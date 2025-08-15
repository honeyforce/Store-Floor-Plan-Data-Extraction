import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

# If Tesseract isn't on PATH (Windows), set it here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="Clip‑Strip Extractor", page_icon="🟥", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #F0F0F0;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        width: 10em;
        border-radius: 12px;
        font-weight: bold;
    }
    .stFileUploader>div>div>input {
        border-radius: 12px;
        padding: 0.5em;
    }
    .stProgress>div>div>div>div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🟥 Clip‑Strip Extractor — Modern Interface")
st.caption("Upload an image with red rectangles and an Excel file. OCR extracts bay codes and matches to Excel.")

# -------------------- Helpers --------------------
LOC_RE = re.compile(r"\d+-[RL]-\d+")

def parse_location(loc: str):
    if not isinstance(loc, str): return (10**9, 10**9, 10**9)
    m = re.match(r'^\s*(\d+)\s*-\s*([LR])\s*-\s*(\d+)\s*$', loc.upper())
    if not m: return (10**9, 10**9, 10**9)
    aisle = int(m.group(1)); side = 0 if m.group(2) == "L" else 1; bay = int(m.group(3))
    return (aisle, side, bay)

def sort_df(df):
    keys = df["Location"].apply(parse_location)
    df = df.assign(_a=[k[0] for k in keys], _s=[k[1] for k in keys], _b=[k[2] for k in keys])
    return df.sort_values(by=["_a","_s","_b"], kind="mergesort").drop(columns=["_a","_s","_b"]).reset_index(drop=True)

def norm_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[\s_]+","",str(c)).upper() for c in df.columns]
    return df

def norm_export_loc(s: str):
    s = str(s).upper().strip()
    s = re.sub(r"\s+","",s).replace("–","-").replace("—","-")
    m = re.match(r"^(\d-[LR])-(?:BAY)?0*(\d+)$", s)
    return f"{m.group(1)}-{m.group(2)}" if m else s

def pick_col(df, candidates):
    for name in candidates:
        k = re.sub(r"[\s_]+","",name).upper()
        if k in df.columns: return k
    for c in df.columns:
        for name in candidates:
            if re.sub(r"[\s_]+","",name).upper() in c:
                return c
    return None

# -------------------- Excel Loader --------------------
def load_excel_tables(xls_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    export_sheet = None; pref_sheet = None
    for s in xls.sheet_names:
        sl = s.lower()
        if export_sheet is None and "export" in sl: export_sheet = s
        if pref_sheet   is None and "prefer" in sl: pref_sheet   = s
    export_sheet = export_sheet or xls.sheet_names[0]
    pref_sheet   = pref_sheet   or xls.sheet_names[-1]

    # Export
    dfe_try = pd.read_excel(xls, sheet_name=export_sheet, header=0)
    dfe = norm_cols(dfe_try)
    loc_col = pick_col(dfe, ["LOCATION","LOC","LOCATIONCODE","BAYLOCATION","LOCN"])
    adj_col = pick_col(dfe, ["ADJACENCY","SECTION","CATEGORY","ADJ"])
    if loc_col is None or adj_col is None:
        dfe_raw = pd.read_excel(xls, sheet_name=export_sheet, header=None)
        def looks_loc(x):
            t = re.sub(r"\s+","",str(x).upper().replace("—","-").replace("–","-"))
            return bool(re.match(r"^\d+-[LR]-(?:BAY)?\d{1,2}$", t))
        scores = [(sum(looks_loc(v) for v in dfe_raw[c].head(40)), c) for c in dfe_raw.columns]
        scores.sort(reverse=True)
        loc_c = scores[0][1]
        others = [c for c in dfe_raw.columns if c != loc_c]
        def avg_len(c): 
            vals = dfe_raw[c].head(40).astype(str).tolist()
            return float(np.mean([len(v) for v in vals])) if vals else 0.0
        adj_c = max(others, key=avg_len) if others else loc_c
        dfe = dfe_raw.rename(columns={loc_c:"LOCATION", adj_c:"ADJACENCY"})[["LOCATION","ADJACENCY"]]
    else:
        dfe = dfe.rename(columns={loc_col:"LOCATION", adj_col:"ADJACENCY"})[["LOCATION","ADJACENCY"]]
    dfe["LocNorm"] = dfe["LOCATION"].apply(norm_export_loc)

    # Preferred
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
    return dfe, dfp

# -------------------- Red detection --------------------
def detect_red_contours(bgr, min_area=100):
    def hsv_mask(img, tol=10, smin=80, vmin=80):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo1 = np.array([0, smin, vmin], np.uint8)
        hi1 = np.array([tol, 255, 255], np.uint8)
        lo2 = np.array([180-tol, smin, vmin], np.uint8)
        hi2 = np.array([179, 255, 255], np.uint8)
        return cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    def rgb_mask(img, thr=35):
        b, g, r = cv2.split(img.astype(np.int16))
        return ((r > g + thr) & (r > b + thr)).astype(np.uint8) * 255
    def contours(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) >= min_area]
    for mask_func in [lambda img: hsv_mask(img,10,80,80),
                      lambda img: hsv_mask(img,16,60,60),
                      rgb_mask]:
        c = contours(mask_func(bgr))
        if c: return c
    return []

# -------------------- OCR --------------------
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
            t = re.sub(r"\s+","",txt.upper().replace("—","-").replace("–","-"))
            m = LOC_RE.search(t)
            if m: return m.group()
    return None

def ocr_codes_from_rect(crop_bgr):
    code = ocr_code_from_crop(crop_bgr)
    return [code] if code else []

# -------------------- Extraction --------------------
def extract_from_image(img_bytes, xls_bytes):
    progress = st.progress(0)
    status_text = st.empty()

    # Load Excel
    status_text.text("Loading Excel...")
    df_export, df_pref = load_excel_tables(xls_bytes)
    progress.progress(10)

    # Decode image
    status_text.text("Decoding image...")
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None: raise RuntimeError("Could not decode image.")
    progress.progress(20)

    # Detect red rectangles
    status_text.text("Detecting red rectangles...")
    cnts = detect_red_contours(bgr)
    progress.progress(40)

    results = []
    overlay = bgr.copy()
    total = len(cnts)
    for i, c in enumerate(cnts, start=1):
        status_text.text(f"OCR on rectangle {i}/{total}...")
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        cv2.drawContours(mask,[c],-1,255,thickness=cv2.FILLED)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
        mask = cv2.erode(mask,k,1)
        ys,xs = np.where(mask>0)
        if len(xs)==0: continue
        x1,x2 = int(xs.min()), int(xs.max())
        y1,y2 = int(ys.min()), int(ys.max())
        crop = bgr[y1:y2+1, x1:x2+1]

        codes = ocr_codes_from_rect(crop)
        if not codes:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,22,33),2)
            cv2.putText(overlay,f"#{i}: (no code)",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,22,33),2,cv2.LINE_AA)
            continue

        for code in codes:
            row = df_export[df_export["LocNorm"]==code]
            adjacency=None
            if not row.empty: adjacency=row.iloc[0]["ADJACENCY"]
            preferred=second=third=None
            if adjacency is not None:
                pr = df_pref[df_pref["_SECTION_NORM"]==str(adjacency).upper().strip()]
                if not pr.empty:
                    preferred = pr.iloc[0]["PREFERRED"]
                    second    = pr.iloc[0]["2ND"]
                    third     = pr.iloc[0]["3RD"]
            results.append({"Location":code,"Adjacency":adjacency,"Preferred":preferred,"2nd":second,"3rd":third})

        label = ", ".join(codes)
        cv2.rectangle(overlay,(x,y),(x+w,y+h),(232,23,255),2)
        cv2.putText(overlay,label,(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(232,23,255),2,cv2.LINE_AA)
        progress.progress(40 + int(50*i/total))

    # Output DataFrame
    status_text.text("Building output Excel...")
    df = pd.DataFrame(results, columns=["Location","Adjacency","Preferred","2nd","3rd"])
    if not df.empty:
        df = df.drop_duplicates(subset=["Location"])
        df = sort_df(df)
        df.insert(0,"No",range(1,len(df)+1))
    else:
        df = pd.DataFrame(columns=["No","Location","Adjacency","Preferred","2nd","3rd"])
    progress.progress(90)

    # Prepare Excel
    status_text.text("Preparing Excel for download...")
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer,index=False,sheet_name="Matched")
    out_buf.seek(0)
    progress.progress(100)
    status_text.text("Done!")

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return df, out_buf, rgb

# -------------------- UI --------------------
st.sidebar.header("📂 Upload Files")
excel_file = st.sidebar.file_uploader("Excel (Export + Preferred)", type=["xlsx"])
img_file   = st.sidebar.file_uploader("Image (PNG/JPG with red rectangles)", type=["png","jpg","jpeg"])

if st.sidebar.button("Process", type="primary", disabled=not(excel_file and img_file)):
    try:
        df_out, excel_buf, preview_rgb = extract_from_image(img_file.read(), excel_file.read())
        if df_out.empty:
            st.warning("No bay codes extracted. Make sure codes like '2-L-15' are fully visible inside each red rectangle.")
        st.image(preview_rgb, caption="Preview (pink = extracted, dark = no code)", use_container_width=True)
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.download_button("Download Excel", data=excel_buf,
                           file_name="matched_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Failed: {e}")
