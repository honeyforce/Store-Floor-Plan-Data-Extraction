import io
import os
import re
import json
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
import camelot
import tempfile

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="Clip-Strip Extractor", page_icon="🟥", layout="wide")
st.markdown("""
    <style>
    .stApp {background-color: #0E1117; color: #F0F0F0;}
    .stButton>button {background-color: #FF4B4B; color: white; height: 3em; width: 10em; border-radius: 12px; font-weight: bold;}
    .stFileUploader>div>div>input {border-radius: 12px; padding: 0.5em;}
    .stProgress>div>div>div>div {background-color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)
st.title("🟥 Clip-Strip Extractor — PDF + Preferred Excel")
st.caption("Upload Floor Plan Image, Bay Info PDF, and (optionally) Preferred Products Excel. The app remembers your Preferred Excel in JSON for future runs.")

PREF_JSON = "preferred_products.json"

# -------------------- Helpers --------------------
LOC_RE = re.compile(r"\d+-[RL]-\d+")

def parse_location(loc: str):
    if not isinstance(loc, str): return (10**9, 10**9, 10**9)
    m = re.match(r'^\s*(\d+)\s*-\s*([LR])\s*-\s*(\d+)\s*$', str(loc).upper())
    if not m: return (10**9, 10**9, 10**9)
    aisle = int(m.group(1))
    side = 0 if m.group(2) == "L" else 1
    bay = int(m.group(3))
    return (aisle, side, bay)

def sort_df(df):
    if df.empty: return df
    keys = df["Location"].apply(parse_location)
    df = df.assign(_a=[k[0] for k in keys], _s=[k[1] for k in keys], _b=[k[2] for k in keys])
    return df.sort_values(by=["_a","_s","_b"], kind="mergesort").drop(columns=["_a","_s","_b"]).reset_index(drop=True)

def norm_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[\s_]+","",str(c)).upper() for c in df.columns]
    return df

def pick_col(df, candidates):
    for name in candidates:
        k = re.sub(r"[\s_]+","",name).upper()
        if k in df.columns: return k
    for c in df.columns:
        for name in candidates:
            if re.sub(r"[\s_]+","",name).upper() in c:
                return c
    return None

# -------------------- Preferred Excel JSON persistence --------------------
def _ensure_preferred_shape(df):
    """Make sure preferred df has SECTION,PREFERRED,2ND,3RD and _SECTION_NORM."""
    if df is None or df.empty:
        return df
    # Normalize column names (but keep originals for values)
    df_norm = df.copy()
    cols_up = {c: re.sub(r"[\s_]+","",str(c)).upper() for c in df_norm.columns}
    inv = {v:k for k,v in cols_up.items()}

    # Map likely names
    sec_name = inv.get("SECTION") or inv.get("CATEGORY") or inv.get("DEPARTMENT") or list(df_norm.columns)[0]
    pref_name = inv.get("PREFERRED", None)
    snd_name  = inv.get("2ND", None)
    trd_name  = inv.get("3RD", None)

    out = pd.DataFrame()
    out["SECTION"]   = df_norm[sec_name].astype(str) if sec_name in df_norm.columns else df_norm.iloc[:,0].astype(str)
    out["PREFERRED"] = df_norm[pref_name] if (pref_name and pref_name in df_norm.columns) else None
    out["2ND"]       = df_norm[snd_name]  if (snd_name  and snd_name  in df_norm.columns) else None
    out["3RD"]       = df_norm[trd_name]  if (trd_name  and trd_name  in df_norm.columns) else None

    out["_SECTION_NORM"] = out["SECTION"].astype(str).str.upper().str.strip()
    return out

def load_preferred_json():
    if os.path.exists(PREF_JSON):
        with open(PREF_JSON,"r",encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return _ensure_preferred_shape(df)
    return None

def save_preferred_json(df):
    df = _ensure_preferred_shape(df)
    if df is not None:
        df.to_json(PREF_JSON, orient="records", force_ascii=False)

def load_preferred_excel(xls_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    pref_sheet = None
    for s in xls.sheet_names:
        if "prefer" in s.lower():
            pref_sheet = s; break
    pref_sheet = pref_sheet or xls.sheet_names[-1]

    dfp_try = pd.read_excel(xls, sheet_name=pref_sheet, header=0)
    dfp = norm_cols(dfp_try)
    if not any(c in dfp.columns for c in ("SECTION","CATEGORY","DEPARTMENT")):
        dfp_raw = pd.read_excel(xls, sheet_name=pref_sheet, header=None)
        while dfp_raw.shape[1] < 4:
            dfp_raw[dfp_raw.shape[1]] = None
        dfp_raw.columns = ["SECTION","PREFERRED","2ND","3RD"] + [f"X{i}" for i in range(4, dfp_raw.shape[1])]
        dfp = norm_cols(dfp_raw[["SECTION","PREFERRED","2ND","3RD"]])
    sec_col = pick_col(dfp, ["SECTION","CATEGORY","DEPARTMENT"]) or dfp.columns[0]
    if sec_col != "SECTION": dfp.rename(columns={sec_col:"SECTION"}, inplace=True)
    for c in ["PREFERRED","2ND","3RD"]:
        if c not in dfp.columns: dfp[c] = None
    dfp["_SECTION_NORM"] = dfp["SECTION"].astype(str).str.upper().str.strip()
    # Return in canonical shape
    return _ensure_preferred_shape(dfp)

# -------------------- PDF → DataFrame (clean) --------------------
def _write_tmp_pdf(pdf_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush(); tmp.close()
    return tmp.name

def load_pdf_to_df(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Read the Bay Info PDF, clean header rows, strip 'PLANOGRAM LISTING' rows,
    normalize headers (remove trailing dots), and compute CUM_SIZE per (AISLE NO, AISLE SIDE).
    """
    tmp_path = _write_tmp_pdf(pdf_bytes)
    try:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream", strip_text='\n')
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    if not tables:
        raise RuntimeError("No table data extracted from PDF.")

    df = pd.concat([t.df for t in tables if not t.df.empty], ignore_index=True)

    # Remove first row if all numeric (0,1,2,...)
    if len(df) > 0 and df.iloc[0].apply(lambda x: str(x).isdigit()).all():
        df = df.iloc[1:].reset_index(drop=True)

    # Remove any rows containing "PLANOGRAM LISTING" (may appear on each page)
    df = df[~df.apply(lambda row: row.astype(str).str.upper().str.contains("PLANOGRAM LISTING").any(), axis=1)]
    df = df.reset_index(drop=True)

    # Use first remaining row as header
    df.columns = [str(c).strip().replace("\n"," ").replace("\r"," ").upper() for c in df.iloc[0]]
    df = df[1:].reset_index(drop=True)

    # Remove trailing dots in headers (e.g., "AISLE NO.")
    df.columns = [c.rstrip('.') for c in df.columns]

    # Flexible rename (handles size/Size, AISLE NO./AISLE NO, etc.)
    cols_map = {}
    for c in df.columns:
        key = c.replace(" ", "").upper()
        if "PLANOGRAM" in key and "NAME" in key:
            cols_map[c] = "PLANOGRAM NAME"
        elif "AISLE" in key and "NO" in key:
            cols_map[c] = "AISLE NO"
        elif "SIDE" in key:
            cols_map[c] = "AISLE SIDE"
        elif "SIZE" in key:
            cols_map[c] = "SIZE"
    df = df.rename(columns=cols_map)

    required = ["PLANOGRAM NAME","AISLE NO","AISLE SIDE","SIZE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"PDF missing required columns: {missing}")

    # Normalize types
    df["AISLE NO"] = pd.to_numeric(df["AISLE NO"], errors="coerce")
    df["SIZE"]     = pd.to_numeric(df["SIZE"], errors="coerce")
    df["AISLE SIDE"] = df["AISLE SIDE"].astype(str).str.strip().str.upper().str[0]  # 'L'/'R'

    # Drop any rows with missing aisle/side/size after coercion
    df = df.dropna(subset=["AISLE NO","AISLE SIDE","SIZE"]).reset_index(drop=True)

    # Compute cumulative size per aisle+side
    df["CUM_SIZE"] = df.groupby(["AISLE NO","AISLE SIDE"])["SIZE"].cumsum()

    return df

# -------------------- OCR helpers --------------------
def remove_vertical_lines(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 12)))
    vert_mask = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vert_mask, connectivity=8)
    keep = np.ones_like(vert_mask)*255
    for i in range(1,num):
        x,y,wcc,hcc,area = stats[i]
        if hcc >= int(0.6*h): keep[labels==i]=0
    cleaned_gray = cv2.bitwise_and(gray, gray, mask=keep)
    return cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)

def preprocess_variants(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants.append(th)
    variants.append(cv2.bitwise_not(th))
    th_adapt = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
    variants.append(th_adapt)
    return variants

def ocr_code_from_crop(crop_bgr):
    crop_bgr = remove_vertical_lines(crop_bgr)
    cfgs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=RL0123456789-",
        "--oem 3 --psm 6 -c tessedit_char_whitelist=RL0123456789-",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=RL0123456789-"
    ]
    for var in preprocess_variants(crop_bgr):
        for cfg in cfgs:
            txt = pytesseract.image_to_string(var, config=cfg)
            t = re.sub(r"\s+","",str(txt).upper().replace("—","-").replace("–","-"))
            m = LOC_RE.search(t)
            if m: return m.group()
    return None

def ocr_codes_from_rect(crop_bgr):
    code = ocr_code_from_crop(crop_bgr)
    return [code] if code else []

# -------------------- Color detection (red & green) --------------------
def detect_colored_contours(bgr, color="red", min_area=100):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if color == "red":
        # red wraps HSV hue, so use two ranges
        lower1 = np.array([0, 80, 80]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 80, 80]); upper2 = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        draw_color = (232, 23, 255)  # magenta/pink
    else:
        # green
        lower = np.array([40, 80, 80]); upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        draw_color = (0, 255, 0)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return cnts, draw_color

# -------------------- Matching helpers --------------------
def match_planogram_for_bay(df_pdf: pd.DataFrame, aisle: int, side: str, bay_number: int):
    row = df_pdf[
        (df_pdf["AISLE NO"] == aisle) &
        (df_pdf["AISLE SIDE"] == side.upper()) &
        (df_pdf["CUM_SIZE"] >= bay_number)
    ].sort_values("CUM_SIZE").head(1)
    if row.empty: return None
    return row.iloc[0]["PLANOGRAM NAME"]

def attach_preferred(preferred_df: pd.DataFrame, adjacency: str):
    preferred = second = third = None
    if adjacency is not None and preferred_df is not None and not preferred_df.empty:
        pr = preferred_df[preferred_df["_SECTION_NORM"]==str(adjacency).upper().strip()]
        if not pr.empty:
            preferred = pr.iloc[0]["PREFERRED"]
            second    = pr.iloc[0]["2ND"]
            third     = pr.iloc[0]["3RD"]
    return preferred, second, third

# -------------------- Main extraction --------------------
def extract_from_image(img_bytes, pdf_bytes, df_pref):
    progress = st.progress(0)
    status_text = st.empty()

    # 1) Load and clean PDF to table
    status_text.text("Loading bay info from PDF…")
    df_pdf = load_pdf_to_df(pdf_bytes)
    progress.progress(15)

    # 2) Decode image
    status_text.text("Decoding floor plan image…")
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None: raise RuntimeError("Could not decode image.")
    overlay = bgr.copy()
    progress.progress(30)

    # 3) OCR from red & green rectangles
    status_text.text("Detecting red and green rectangles + OCR…")
    results = []
    for color in ("red","green"):
        cnts, draw_color = detect_colored_contours(bgr, color=color)
        for i, c in enumerate(cnts, start=1):
            x,y,w,h = cv2.boundingRect(c)
            crop = bgr[y:y+h, x:x+w]
            codes = ocr_codes_from_rect(crop)

            if not codes:
                cv2.rectangle(overlay,(x,y),(x+w,y+h),(33,33,33),2)
                cv2.putText(overlay, f"(no code)", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33,33,33), 1, cv2.LINE_AA)
                continue

            for code in codes:
                try:
                    aisle_str, side, bay_str = code.split("-")
                    aisle = int(aisle_str); bay_number = int(bay_str); side = side.upper()
                except Exception:
                    continue

                adjacency = match_planogram_for_bay(df_pdf, aisle, side, bay_number)
                preferred, second, third = attach_preferred(df_pref, adjacency)
                results.append({"Location": code, "Adjacency": adjacency, "Preferred": preferred, "2nd": second, "3rd": third})

            label = ", ".join(codes) if codes else "(no code)"
            cv2.rectangle(overlay,(x,y),(x+w,y+h), draw_color, 2)
            cv2.putText(overlay, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2, cv2.LINE_AA)

    progress.progress(60)

    # 4) Add FIRST and LAST bay per (AISLE NO, AISLE SIDE)
    status_text.text("Adding first (…-…-1) and last bay per aisle/side…")
    first_last_rows = []
    for (aisle, side), group in df_pdf.groupby(["AISLE NO","AISLE SIDE"]):
        if group.empty: continue
        side = str(side).upper()
        group = group.sort_values("CUM_SIZE")  # ensure order

        # First bay is always 1
        first_bay_num = 1
        # Last bay uses final cumulative size (rounded)
        last_cum = group["CUM_SIZE"].iloc[-1]
        last_bay_num = int(round(float(last_cum))) if pd.notna(last_cum) else None

        first_adj = group.iloc[0]["PLANOGRAM NAME"]
        last_adj  = group.iloc[-1]["PLANOGRAM NAME"] if last_bay_num is not None else None

        first_loc = f"{int(aisle)}-{side}-{first_bay_num}"
        last_loc  = f"{int(aisle)}-{side}-{last_bay_num}" if last_bay_num is not None and last_bay_num >= first_bay_num else None

        p1, s1, t1 = attach_preferred(df_pref, first_adj)
        first_last_rows.append({"Location": first_loc, "Adjacency": first_adj, "Preferred": p1, "2nd": s1, "3rd": t1})
        if last_loc:
            p2, s2, t2 = attach_preferred(df_pref, last_adj)
            first_last_rows.append({"Location": last_loc, "Adjacency": last_adj, "Preferred": p2, "2nd": s2, "3rd": t2})

    progress.progress(75)

    # 5) Merge results (OCR + first/last), de-dup, sort
    status_text.text("Merging OCR results with first/last bays…")
    df_res_ocr = pd.DataFrame(results, columns=["Location","Adjacency","Preferred","2nd","3rd"])
    df_res_fl  = pd.DataFrame(first_last_rows, columns=["Location","Adjacency","Preferred","2nd","3rd"])
    df_out = pd.concat([df_res_ocr, df_res_fl], ignore_index=True)
    if not df_out.empty:
        df_out = df_out.drop_duplicates(subset=["Location"], keep="first")
        df_out = sort_df(df_out)
        df_out.insert(0, "No", range(1, len(df_out) + 1))
    else:
        df_out = pd.DataFrame(columns=["No","Location","Adjacency","Preferred","2nd","3rd"])
    progress.progress(90)

    # 6) Build Excel
    status_text.text("Preparing Excel for download…")
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Matched")
    out_buf.seek(0)
    progress.progress(100)
    status_text.text("Done!")

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return df_out, out_buf, rgb, df_pdf

# -------------------- Sidebar Uploads --------------------
st.sidebar.header("📂 Upload Files")
pref_file = st.sidebar.file_uploader("Preferred Products Excel", type=["xlsx"])
pdf_file  = st.sidebar.file_uploader("Bay Info PDF", type=["pdf"])
img_file  = st.sidebar.file_uploader("Floor Plan Image", type=["png","jpg","jpeg"])

# Preferred: use uploaded Excel if present; else fall back to JSON memory
df_pref = None
if pref_file:
    df_pref = load_preferred_excel(pref_file.read())
    save_preferred_json(df_pref)
    st.sidebar.success("Preferred Products updated & saved.")
else:
    df_pref = load_preferred_json()

# -------------------- Main Action --------------------
if st.sidebar.button("Process"):
    if not (pdf_file and img_file):
        st.warning("Upload Bay Info PDF and Floor Plan Image first!")
    elif df_pref is None or df_pref.empty:
        st.warning("Preferred Products are missing. Upload a Preferred Excel at least once so it can be saved to JSON.")
    else:
        df_out, excel_buf, preview_rgb, df_pdf = extract_from_image(
            img_file.read(), pdf_file.read(), df_pref
        )
        st.image(preview_rgb, caption="Preview (magenta=red rectangles, green=green rectangles)", use_container_width=True)
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.download_button("Download Excel", data=excel_buf,
                           file_name="matched_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
