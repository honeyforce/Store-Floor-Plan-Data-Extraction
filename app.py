import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

# OPTIONAL (Windows): set Tesseract path if not on PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Clip-Strip Extractor", page_icon="🟨", layout="centered")
st.title("🟨 Clip-Strip Location Extractor (Dot-inside-Box)")
st.caption("Upload a floor plan image + store Excel, set dot color, and get Location → Adjacency → Preferred (2nd/3rd).")

# ===================== Helpers =====================

def hex_to_rgb(hex_str: str):
    hs = hex_str.strip().lstrip('#')
    if len(hs) == 3:
        hs = ''.join([c*2 for c in hs])
    r = int(hs[0:2], 16)
    g = int(hs[2:4], 16)
    b = int(hs[4:6], 16)
    return (r, g, b)

def rgb_to_hsv_range(rgb, tol_h=18, tol_s=120, tol_v=120):
    """OpenCV HSV: H 0..179, S/V 0..255. Build a +/- tolerance range from a given RGB."""
    bgr = np.uint8([[list(reversed(rgb))]])  # RGB→BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    H, S, V = hsv.tolist()
    low  = np.array([max(0,   H - tol_h), max(0,   S - tol_s), max(0,   V - tol_v)], dtype=np.uint8)
    high = np.array([min(179, H + tol_h), min(255, S + tol_s), min(255, V + tol_v)], dtype=np.uint8)
    return low, high

def find_yellow_dots(hsv_img, lower_hsv, upper_hsv):
    """Detect yellow dots by color; return a list of dicts with box + center."""
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        # filter for small-ish dot marks; adjust if your dot size differs
        if 20 <= area <= 2500 and 5 <= w <= 70 and 5 <= h <= 70:
            cx, cy = x + w//2, y + h//2
            dots.append({"box": (x,y,w,h), "center": (cx,cy)})
    # stable reading order: roughly by rows, then x
    dots.sort(key=lambda d: (d["center"][1]//80, d["center"][0]))
    return dots

def get_nearest_text_around(bgr_img, dot_center, search_frac=(0.25, 0.18)):
    """
    Find nearest text line around the dot using Tesseract word boxes.
    search_frac: fraction of (W,H) for a local window centered on the dot.
    Returns (line_text, debug_words).
    """
    H, W = bgr_img.shape[:2]
    cx, cy = dot_center

    win_w = max(120, int(W * search_frac[0]))
    win_h = max(80,  int(H * search_frac[1]))
    x1 = max(0, cx - win_w // 2)
    y1 = max(0, cy - win_h // 2)
    x2 = min(W, x1 + win_w)
    y2 = min(H, y1 + win_h)
    roi = bgr_img[y1:y2, x1:x2]

    # We need word coordinates, so no whitelist here
    data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DATAFRAME,
                                     config="--oem 3 --psm 6")
    if data is None or data.empty:
        return None, []

    data = data.dropna(subset=["text"])
    # keep words with reasonable confidence
    try:
        data = data[data["conf"].astype(float) > 45].copy()
    except Exception:
        data = data.copy()
    if data.empty:
        return None, []

    # Absolute coords
    data["abs_left"] = data["left"] + x1
    data["abs_top"] = data["top"] + y1
    data["abs_right"] = data["abs_left"] + data["width"]
    data["abs_bottom"] = data["abs_top"] + data["height"]
    data["abs_cx"] = data["abs_left"] + data["width"]/2.0
    data["abs_cy"] = data["abs_top"]  + data["height"]/2.0

    # Group words by (block,line) and score lines by proximity
    best_key, best_score = None, 1e12
    for (block, line), grp in data.groupby(["block_num", "line_num"]):
        dx = (grp["abs_cx"] - cx).abs().min()
        dy = (grp["abs_cy"] - cy).abs().min()
        score = (dx**2 + (0.6*dy)**2) ** 0.5   # weight horizontal more than vertical
        if score < best_score:
            best_score = score
            best_key = (block, line)

    if best_key is None:
        return None, []

    best_line = data[(data["block_num"] == best_key[0]) & (data["line_num"] == best_key[1])] \
                .sort_values(by="abs_left")

    words = [w for w in best_line["text"].tolist() if isinstance(w, str) and w.strip()]
    debug_words = best_line[["abs_left","abs_top","width","height","text"]].to_dict("records")
    line_text = " ".join(words).strip() if words else None
    return line_text, debug_words

def extract_location_from_text(s):
    """Normalize any text to a location like '7-R-18' (handles minor OCR issues)."""
    if not s:
        return None
    t = s.upper().replace("—","-").replace("–","-")
    t = re.sub(r'\s+', '', t)
    t = t.replace('I','1')  # common OCR confusion
    m = re.search(r'(\d-[RL]-\d{1,2})', t)
    return m.group(1) if m else None

def norm_export_loc(s: str):
    """Normalize Excel 'Location' (strip BAY, spaces, fancy dashes, leading zeros)."""
    s = str(s).upper().strip()
    s = re.sub(r'\s+', '', s)
    s = s.replace('–','-').replace('—','-')
    m = re.match(r'(\d-[LR])-(?:BAY)?0*(\d+)$', s)
    return f"{m.group(1)}-{m.group(2)}" if m else s

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

def tolerant_lookup(df_export: pd.DataFrame, loc_norm: str):
    """Exact match → contains() fallback."""
    r = df_export[df_export["LocNorm"] == loc_norm]
    if not r.empty: return r.iloc[0]
    r = df_export[df_export["LocNorm"].str.contains(re.escape(loc_norm), na=False)]
    if not r.empty: return r.iloc[0]
    return None

# ===================== Pipeline =====================

def run_pipeline(image_bytes: bytes, excel_bytes: bytes,
                 hex_color: str | None, rgb_color: str | None,
                 tol_h: int, tol_s: int, tol_v: int,
                 progress_cb=lambda pct, msg: None):
    # 1) Color range
    if hex_color and hex_color.strip():
        rgb = hex_to_rgb(hex_color)
    elif rgb_color and rgb_color.strip():
        parts = [int(x) for x in rgb_color.split(',')]
        if len(parts) != 3: raise ValueError("RGB must be 'R,G,B'")
        rgb = tuple(parts)
    else:
        raise ValueError("Provide a HEX or RGB dot color.")
    lower_hsv, upper_hsv = rgb_to_hsv_range(rgb, tol_h, tol_s, tol_v)

    # 2) Load Excel (export + preferred)
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
    dots = find_yellow_dots(hsv, lower_hsv, upper_hsv)
    progress_cb(45, f"Found {len(dots)} dots.")

    # 5) For each dot, read nearest text around it
    progress_cb(55, f"OCR around {len(dots)} dots…")
    out_rows, failures, samples = [], [], []
    total = len(dots) or 1

    for i, d in enumerate(dots, start=1):
        (cx, cy) = d["center"]
        line_text, dbg = get_nearest_text_around(bgr, (cx, cy))
        loc_code = extract_location_from_text(line_text)

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
                failures.append({"dot": i, "reason": "No Excel match", "ocr_line": line_text})
        else:
            failures.append({"dot": i, "reason": "OCR empty/line not found", "ocr_line": line_text})

        out_rows.append({
            "Location": loc_code,
            "Adjacency": adjacency,
            "Preferred": preferred,
            "2nd": second,
            "3rd": third
        })

        if len(samples) < 5:
            x,y,w,h = d["box"]
            pad = 24
            x1 = max(0, x-pad); y1=max(0,y-pad)
            x2 = min(bgr.shape[1], x+w+pad); y2 = min(bgr.shape[0], y+h+pad)
            crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            samples.append((crop, loc_code or "(none)", line_text or ""))

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
                    st.warning(f"Unmatched/empty: {len(fails)}")
                    st.dataframe(pd.DataFrame(fails))
                else:
                    st.info("No failures recorded.")
                if diag["samples"]:
                    st.caption("Sample dot crops (first 5):")
                    for rgb_img, loc_txt, raw_line in diag["samples"]:
                        st.image(rgb_img, caption=f"OCR → {loc_txt} | line: {raw_line}", use_container_width=False)
        except Exception as e:
            st.error(f"Failed: {e}")
