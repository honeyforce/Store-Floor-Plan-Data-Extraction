import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

# If Tesseract isn't on PATH (Windows), uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Clip-Strip Extractor (Red Outline)", page_icon="🟥", layout="centered")
st.title("🟥 Clip-Strip Extractor — Red Outline → Interior OCR")
st.caption("Upload a floor plan image + store Excel. The app detects RED outlines, reads text inside, and matches to Excel.")

# ---------- Helpers ----------
def norm_export_loc(s: str):
    """Normalize Excel 'Location' to A-S-## (strip BAY, spaces, fancy dashes, zeros)."""
    s = str(s).upper().strip()
    s = re.sub(r'\s+', '', s)
    s = s.replace('–','-').replace('—','-')
    m = re.match(r'(\d-[LR])-(?:BAY)?0*(\d+)$', s)
    return f"{m.group(1)}-{m.group(2)}" if m else s

def tolerant_lookup(df_export: pd.DataFrame, loc_norm: str):
    r = df_export[df_export["LocNorm"] == loc_norm]
    if not r.empty: return r.iloc[0]
    r = df_export[df_export["LocNorm"].str.contains(re.escape(loc_norm), na=False)]
    if not r.empty: return r.iloc[0]
    return None

def find_sheets(xls: pd.ExcelFile):
    export_sheet = None; pref_sheet = None
    for s in xls.sheet_names:
        sl = s.lower()
        if export_sheet is None and "export" in sl:   export_sheet = s
        if pref_sheet   is None and "prefer" in sl:   pref_sheet   = s
    export_sheet = export_sheet or xls.sheet_names[0]
    pref_sheet   = pref_sheet   or xls.sheet_names[-1]
    return export_sheet, pref_sheet

def extract_location_from_text(s):
    """Return normalized 'A-S-##' from OCR text (handle common OCR quirks)."""
    if not s: return None
    t = s.upper().replace("—","-").replace("–","-")
    t = re.sub(r'\s+', '', t).replace('I','1')
    m = re.search(r'(\d-[RL]-\d{1,2})', t)
    return m.group(1) if m else None

# ---------- Red-outline detection ----------
def red_hsv_ranges(hue_tol=10, sat_tol=120, val_tol=120):
    """
    Red wraps around the HSV hue circle (0 and 180). We build two ranges:
    low reds (~0) and high reds (~180).
    """
    # Base red in OpenCV HSV is around H≈0 and H≈179.
    # We'll cover both with two masks.
    lo1 = np.array([0,            max(0, 200 - sat_tol), max(0, 200 - val_tol)], dtype=np.uint8)
    hi1 = np.array([hue_tol,      255,                   255],                   dtype=np.uint8)
    lo2 = np.array([180 - hue_tol, max(0, 200 - sat_tol), max(0, 200 - val_tol)], dtype=np.uint8)
    hi2 = np.array([179,          255,                   255],                   dtype=np.uint8)
    return (lo1, hi1, lo2, hi2)

def detect_red_outlines(hsv_img, hue_tol=10, sat_tol=120, val_tol=120):
    lo1, hi1, lo2, hi2 = red_hsv_ranges(hue_tol, sat_tol, val_tol)
    mask1 = cv2.inRange(hsv_img, lo1, hi1)
    mask2 = cv2.inRange(hsv_img, lo2, hi2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Bridge tiny gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts, mask

def interior_mask_from_outline(contour, shape, erode_px=3):
    """Filled interior from an outline; erode to avoid the colored stroke."""
    h, w = shape[:2]
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        filled = cv2.erode(filled, k, 1)
    return filled

def crop_by_mask(bgr_img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return bgr_img[y1:y2+1, x1:x2+1], (x1, y1, x2-x1+1, y2-y1+1)

# ---------- OCR ----------
def preprocess_box_for_ocr(box_img):
    """OCR-friendly variants from interior region."""
    gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)

    variants = []
    v_otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    v_adap = cv2.adaptiveThreshold(up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,9)
    blur   = cv2.GaussianBlur(up,(0,0),1.0)
    v_shrp = cv2.threshold(cv2.addWeighted(up, 1.9, blur, -0.9, 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    ker = np.ones((2,2), np.uint8)
    v_dil = cv2.dilate(v_otsu, ker, 1)
    v_ero = cv2.erode(v_otsu,  ker, 1)
    base = [v_otsu, v_adap, v_shrp, v_dil, v_ero]
    variants.extend(base)
    variants.extend([cv2.bitwise_not(v) for v in base])
    return variants

def ocr_interior_text(box_img):
    """Return (raw_text, extracted_location)."""
    cfgs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-/ ",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-/ ",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-/ ",
    ]
    best_raw = ""
    for var in preprocess_box_for_ocr(box_img):
        for cfg in cfgs:
            txt = pytesseract.image_to_string(var, config=cfg)
            if len(txt) > len(best_raw):
                best_raw = txt
            loc = extract_location_from_text(txt)
            if loc:
                return txt, loc
    return best_raw, extract_location_from_text(best_raw)

# ---------- Pipeline ----------
def run_pipeline(image_bytes: bytes, excel_bytes: bytes,
                 hue_tol: int, sat_tol: int, val_tol: int,
                 erode_px: int,
                 progress_cb=lambda pct, msg: None):

    # Load Excel
    progress_cb(10, "Loading Excel…")
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    export_sheet, pref_sheet = find_sheets(xls)

    df_export = pd.read_excel(xls, sheet_name=export_sheet)
    df_export.columns = [str(c).strip().title() for c in df_export.columns]
    if "Location" not in df_export.columns or "Adjacency" not in df_export.columns:
        raise RuntimeError(f"'{export_sheet}' must contain 'Location' and 'Adjacency'. Found: {df_export.columns.tolist()}")
    df_export["LocNorm"] = df_export["Location"].apply(norm_export_loc)

    df_pref = pd.read_excel(xls, sheet_name=pref_sheet)
    df_pref.columns = [str(c).strip().upper() for c in df_pref.columns]
    if "SECTION" not in df_pref.columns:
        raise RuntimeError(f"'{pref_sheet}' must contain 'SECTION' (and ideally 'PREFERRED', '2ND', '3RD'). Found: {df_pref.columns.tolist()}")
    df_pref["_SECTION_NORM"] = df_pref["SECTION"].astype(str).str.upper().str.strip()

    # Decode image
    progress_cb(20, "Reading image…")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode the uploaded image.")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Detect red outlines
    progress_cb(35, "Detecting red outlines…")
    contours, mask_outline = detect_red_outlines(hsv, hue_tol, sat_tol, val_tol)
    progress_cb(45, f"Found {len(contours)} outline(s).")

    # OCR each interior
    progress_cb(55, f"OCR interior for {len(contours)} outline(s)…")
    out_rows, failures, samples = [], [], []
    total = len(contours) or 1

    for i, c in enumerate(contours, start=1):
        interior_mask = interior_mask_from_outline(c, bgr.shape, erode_px=erode_px)
        crop, rect = crop_by_mask(bgr, interior_mask)

        raw_text = None
        loc_code = None
        if crop is not None:
            raw_text, loc_code = ocr_interior_text(crop)
        else:
            failures.append({"outline": i, "reason": "Empty interior mask"})

        adjacency = preferred = second = third = None
        if loc_code:
            row = tolerant_lookup(df_export, loc_code)
            if row is not None:
                adjacency = row["Adjacency"]
                pr = df_pref[df_pref["_SECTION_NORM"] == str(adjacency).upper().strip()]
                preferred = pr.iloc[0]["PREFERRED"] if ("PREFERRED" in df_pref.columns and not pr.empty) else None
                second    = pr.iloc[0]["2ND"]       if ("2ND" in df_pref.columns       and not pr.empty) else None
                third     = pr.iloc[0]["3RD"]       if ("3RD" in df_pref.columns       and not pr.empty) else None
            else:
                failures.append({"outline": i, "reason": "No Excel match", "ocr_text": raw_text})
        else:
            failures.append({"outline": i, "reason": "OCR empty / no location", "ocr_text": raw_text})

        out_rows.append({
            "Location": loc_code,
            "Adjacency": adjacency,
            "Preferred": preferred,
            "2nd": second,
            "3rd": third
        })

        if len(samples) < 5:
            if rect is not None:
                x, y, w, h = rect
                pad = 12
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
                crop_dbg = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                cap = f"LOC: {loc_code or '(none)'}"
            else:
                crop_dbg = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cap = "No rect"
            samples.append((crop_dbg, cap, raw_text or ""))

        progress_cb(55 + int(35*i/total), f"OCR & matching… ({i}/{total})")

    # Output Excel
    progress_cb(95, "Preparing output Excel…")
    df_out = pd.DataFrame(out_rows).drop_duplicates(subset=["Location"])
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Matched")
    excel_buf.seek(0)
    progress_cb(100, "Done.")

    diagnostics = {"outlines_found": len(contours), "failures": failures, "samples": samples}
    return df_out, excel_buf, diagnostics

# ---------- UI ----------
with st.form("extract_form"):
    img_file  = st.file_uploader("Floor plan image (PNG/JPG)", type=["png","jpg","jpeg"])
    xlsx_file = st.file_uploader("Store Excel (Export + Preferred sheets)", type=["xlsx"])

    with st.expander("Advanced (optional)"):
        st.write("Red detection defaults are good for pure #FF0000. Tweak only if needed.")
        hue_tol = st.slider("Hue tolerance (±)", 0, 30, 10, 1)
        sat_tol = st.slider("Saturation tolerance (±)", 0, 255, 120, 5)
        val_tol = st.slider("Value tolerance (±)", 0, 255, 120, 5)
        erode_px = st.slider("Interior erosion (px)", 0, 8, 3, 1, help="Move OCR crop away from the red stroke")

    submitted = st.form_submit_button("Process", type="primary")

if submitted:
    if not img_file or not xlsx_file:
        st.error("Please upload both the floor plan image and the store Excel.")
    else:
        prog = st.progress(0)
        status = st.empty()
        def cb(pct, msg): prog.progress(min(pct,100)); status.info(msg)
        try:
            df_out, excel_buf, diag = run_pipeline(
                image_bytes=img_file.read(),
                excel_bytes=xlsx_file.read(),
                hue_tol=int(hue_tol),
                sat_tol=int(sat_tol),
                val_tol=int(val_tol),
                erode_px=int(erode_px),
                progress_cb=cb
            )
            st.success("Processing complete.")
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Excel",
                data=excel_buf,
                file_name="matched_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            with st.expander("Diagnostics"):
                st.write(f"Red outlines detected: {diag['outlines_found']}")
                fails = diag["failures"]
                if fails:
                    st.warning(f"Issues: {len(fails)}")
                    st.dataframe(pd.DataFrame(fails))
                else:
                    st.info("No failures recorded.")
                if diag["samples"]:
                    st.caption("Sample crops (first 5):")
                    for rgb_img, cap, raw in diag["samples"]:
                        st.image(rgb_img, caption=f"{cap} | raw OCR: {raw}", use_container_width=False, hide_index=True)
        except Exception as e:
            st.error(f"Failed: {e}")
