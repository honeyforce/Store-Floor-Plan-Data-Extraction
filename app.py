import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

# OPTIONAL (Windows): set this if Tesseract isn't on PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Clip-Strip Extractor (Dot in Box)", page_icon="🟨", layout="centered")
st.title("🟨 Clip-Strip Extractor — Dot Inside Black Box")
st.caption("Upload a floor plan image + store Excel, give dot color, and get Location → Adjacency → Preferred (2nd/3rd).")

# ===================== Utilities =====================

def hex_to_rgb(hex_str: str):
    hs = hex_str.strip().lstrip('#')
    if len(hs) == 3:
        hs = ''.join([c*2 for c in hs])
    r = int(hs[0:2], 16)
    g = int(hs[2:4], 16)
    b = int(hs[4:6], 16)
    return (r, g, b)

def rgb_to_hsv_range(rgb, tol_h=18, tol_s=120, tol_v=120):
    """OpenCV HSV: H 0..179, S/V 0..255."""
    bgr = np.uint8([[list(reversed(rgb))]])  # RGB→BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    H, S, V = hsv.tolist()
    low  = np.array([max(0,   H - tol_h), max(0,   S - tol_s), max(0,   V - tol_v)], dtype=np.uint8)
    high = np.array([min(179, H + tol_h), min(255, S + tol_s), min(255, V + tol_v)], dtype=np.uint8)
    return low, high

def norm_export_loc(s: str):
    """Normalize Excel 'Location' to A-S-## form (strip BAY, spaces, fancy dashes, zeros)."""
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
        if export_sheet is None and "export" in sl:
            export_sheet = s
        if pref_sheet is None and "prefer" in sl:
            pref_sheet = s
    export_sheet = export_sheet or xls.sheet_names[0]
    pref_sheet   = pref_sheet   or xls.sheet_names[-1]
    return export_sheet, pref_sheet

def extract_location_from_text(s):
    """Return normalized 'A-S-##' from OCR text (handle common OCR quirks)."""
    if not s:
        return None
    t = s.upper().replace("—","-").replace("–","-")
    t = re.sub(r'\s+', '', t)
    t = t.replace('I','1')  # common misread
    m = re.search(r'(\d-[RL]-\d{1,2})', t)
    return m.group(1) if m else None

# ===================== Detection & OCR =====================

def detect_yellow_dots(hsv_img, lower_hsv, upper_hsv):
    """Return list of dot dicts: {'box': (x,y,w,h), 'center': (cx,cy)}."""
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if 20 <= area <= 2500 and 5 <= w <= 70 and 5 <= h <= 70:
            cx, cy = x + w//2, y + h//2
            dots.append({"box": (x,y,w,h), "center": (cx,cy)})
    # Stable reading order: rows then x
    dots.sort(key=lambda d: (d["center"][1]//80, d["center"][0]))
    return dots

def find_black_box_containing_dot(bgr_img, dot_center, search_pad_px=80):
    """
    Find the black rectangular label box that contains the dot.
    1) Make a local search window around the dot.
    2) Detect dark edges/lines; find rectangular contours.
    3) Return bounding rect of the contour that contains the dot.
    """
    H, W = bgr_img.shape[:2]
    cx, cy = dot_center
    x1 = max(0, cx - search_pad_px); y1 = max(0, cy - search_pad_px)
    x2 = min(W, cx + search_pad_px); y2 = min(H, cy + search_pad_px)
    roi = bgr_img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Emphasize black box lines
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Slight blur to join line gaps
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    # Morphological blackhat to highlight dark lines on light background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, kernel)
    # Canny edges on blackhat
    edges = cv2.Canny(blackhat, 30, 120)
    # Close small gaps
    mker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, mker, iterations=1)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = -1
    for c in cnts:
        rx, ry, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if area < 300 or area > (search_pad_px*search_pad_px*3):
            continue
        # approximate polygon for rectangularity
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # Heuristic: 4 corners, mostly rectangular-ish
        if len(approx) < 4 or len(approx) > 8:
            continue

        # Convert to absolute coordinates
        abs_x1, abs_y1 = x1 + rx, y1 + ry
        abs_x2, abs_y2 = abs_x1 + rw, abs_y1 + rh

        # Check if dot is inside this rect
        if not (abs_x1 <= cx <= abs_x2 and abs_y1 <= cy <= abs_y2):
            continue

        # Prefer tighter boxes (smaller area) that still contain the dot
        score = -area
        if score > best_score:
            best_score = score
            best_rect = (abs_x1, abs_y1, rw, rh)

    return best_rect  # (x, y, w, h) in absolute coords

def preprocess_box_for_ocr(box_img):
    """
    Produce a list of OCR-friendly variants from a cropped label box.
    """
    gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
    # Upscale for thin fonts
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

def ocr_box_text(box_img):
    """
    OCR the text inside the label box; return raw and cleaned results.
    We try multiple variants & PSM modes; then extract the location pattern.
    """
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
            # Quick check if location is already present
            loc = extract_location_from_text(txt)
            if loc:
                return txt, loc
    # If not found, still attempt extraction from the best_raw
    return best_raw, extract_location_from_text(best_raw)

# ===================== Pipeline =====================

def run_pipeline(image_bytes: bytes, excel_bytes: bytes,
                 hex_color: str | None, rgb_color: str | None,
                 tol_h: int, tol_s: int, tol_v: int,
                 progress_cb=lambda pct, msg: None):

    # 1) Color → HSV range
    if hex_color and hex_color.strip():
        rgb = hex_to_rgb(hex_color)
    elif rgb_color and rgb_color.strip():
        parts = [int(x) for x in rgb_color.split(',')]
        if len(parts) != 3:
            raise ValueError("RGB must be 'R,G,B'")
        rgb = tuple(parts)
    else:
        raise ValueError("Provide a HEX or RGB dot color.")
    lower_hsv, upper_hsv = rgb_to_hsv_range(rgb, tol_h, tol_s, tol_v)

    # 2) Load Excel
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

    # 3) Decode image
    progress_cb(20, "Reading image…")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode the uploaded image.")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 4) Detect yellow dots
    progress_cb(35, "Detecting yellow dots…")
    dots = detect_yellow_dots(hsv, lower_hsv, upper_hsv)
    progress_cb(45, f"Found {len(dots)} dots.")

    # 5) For each dot, find its black box and OCR only inside that box
    progress_cb(55, f"OCR boxes for {len(dots)} dots…")
    out_rows, failures, samples = [], [], []
    total = len(dots) or 1

    for i, d in enumerate(dots, start=1):
        (cx, cy) = d["center"]
        rect = find_black_box_containing_dot(bgr, (cx, cy), search_pad_px=90)

        raw_text = None
        loc_code = None
        if rect is not None:
            x, y, w, h = rect
            # Slight insets to avoid box border lines in OCR
            inset = max(2, min(w,h)//25)
            xi1 = max(0, x + inset); yi1 = max(0, y + inset)
            xi2 = min(bgr.shape[1], x + w - inset); yi2 = min(bgr.shape[0], y + h - inset)
            box_crop = bgr[yi1:yi2, xi1:xi2]
            raw_text, loc_code = ocr_box_text(box_crop)
        else:
            failures.append({"dot": i, "reason": "No black box found around dot"})

        adjacency = preferred = second = third = None
        if loc_code:
            row = tolerant_lookup(df_export, loc_code)
            if row is not None:
                adjacency = row["Adjacency"]
                pr = df_pref[df_pref["_SECTION_NORM"] == str(adjacency).upper().strip()]
                preferred = pr.iloc[0]["PREFERRED"] if ("PREFERRED" in df_pref.columns and not pr.empty) else None
                second    = pr.iloc[0]["2ND"] if ("2ND" in df_pref.columns and not pr.empty) else None
                third     = pr.iloc[0]["3RD"] if ("3RD" in df_pref.columns and not pr.empty) else None
            else:
                failures.append({"dot": i, "reason": "No Excel match", "ocr_text": raw_text})
        else:
            failures.append({"dot": i, "reason": "OCR empty / no location found", "ocr_text": raw_text})

        out_rows.append({
            "Location": loc_code,
            "Adjacency": adjacency,
            "Preferred": preferred,
            "2nd": second,
            "3rd": third
        })

        if len(samples) < 5:
            # Save a small debug crop around the box (or dot if box not found)
            if rect is not None:
                x, y, w, h = rect
                pad = 16
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
                crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                cap = f"LOC: {loc_code or '(none)'}"
            else:
                x,y,w,h = d["box"]
                pad = 24
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
                crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                cap = "No box found"
            samples.append((crop, cap, raw_text or ""))

        progress_cb(55 + int(35*i/total), f"OCR & matching… ({i}/{total})")

    # 6) Build output Excel (in memory)
    progress_cb(95, "Preparing output Excel…")
    df_out = pd.DataFrame(out_rows).drop_duplicates(subset=["Location"])
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Matched")
    excel_buf.seek(0)
    progress_cb(100, "Done.")

    diagnostics = {"dots_found": len(dots), "failures": failures, "samples": samples}
    return df_out, excel_buf, diagnostics

# ===================== UI =====================

with st.form("extract_form"):
    img_file  = st.file_uploader("Floor plan image (PNG/JPG)", type=["png","jpg","jpeg"])
    xlsx_file = st.file_uploader("Store Excel (Export + Preferred sheets)", type=["xlsx"])
    c1, c2 = st.columns(2)
    with c1:
        hex_color = st.text_input("Dot Color (HEX, e.g. #F3EC6B)", value="#F3EC6B")
    with c2:
        rgb_color = st.text_input("Or RGB (R,G,B)", placeholder="243,236,107")

    c3, c4, c5 = st.columns(3)
    with c3:
        tol_h = st.number_input("Hue tolerance (±)", min_value=0, max_value=60, value=18, step=1)
    with c4:
        tol_s = st.number_input("Sat tolerance (±)", min_value=0, max_value=255, value=120, step=5)
    with c5:
        tol_v = st.number_input("Val tolerance (±)", min_value=0, max_value=255, value=120, step=5)

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
                hex_color=hex_color,
                rgb_color=rgb_color,
                tol_h=int(tol_h), tol_s=int(tol_s), tol_v=int(tol_v),
                progress_cb=cb
            )
            st.success("Processing complete.")
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download Excel",
                data=excel_buf,
                file_name="matched_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            with st.expander("Diagnostics"):
                st.write(f"Dots detected: {diag['dots_found']}")
                fails = diag["failures"]
                if fails:
                    st.warning(f"Issues: {len(fails)}")
                    st.dataframe(pd.DataFrame(fails))
                else:
                    st.info("No failures recorded.")
                if diag["samples"]:
                    st.caption("Sample crops (first 5):")
                    for rgb_img, cap, raw in diag["samples"]:
                        st.image(rgb_img, caption=f"{cap} | raw OCR: {raw}", use_container_width=False)
        except Exception as e:
            st.error(f"Failed: {e}")
