"""
invoice_table_extractor.py

Dependencies (pip):
  pip install opencv-python pillow pytesseract pdf2image pandas numpy regex

System:
  - Install Tesseract OCR (system): https://github.com/tesseract-ocr/tesseract
  - Install Poppler (for pdf2image): e.g. apt-get install poppler-utils

Optional (better accuracy on complex docs):
  pip install layoutparser[detectron2]  # heavy, for ML table detection
  pip install easyocr

Usage:
  from invoice_table_extractor import extract_table_from_invoice
  df = extract_table_from_invoice("/path/to/invoice.png")
  df.to_csv("extracted.csv", index=False)
"""

import os
import re
import math
import tempfile
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract

# Optional imports (catch if not installed)
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except Exception:
    LAYOUTPARSER_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# PDF handling
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# ----------------------
# Utility helpers
# ----------------------
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def read_image(path_or_pil) -> np.ndarray:
    """Load an image from disk or a PIL image and return BGR numpy (cv2)."""
    if isinstance(path_or_pil, Image.Image):
        return pil_to_cv2(path_or_pil)
    path = str(path_or_pil)
    if path.lower().endswith(".pdf"):
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image/poppler required to convert PDFs. Install pdf2image and poppler.")
        pages = convert_from_path(path, dpi=300)
        # choose first page by default
        pil = pages[0]
        return pil_to_cv2(pil)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # try PIL fallback
            pil = Image.open(path).convert("RGB")
            return pil_to_cv2(pil)
        return img

# ----------------------
# Preprocessing + deskew
# ----------------------
def preprocess_for_table(img_bgr: np.ndarray, max_dim=2500) -> np.ndarray:
    """
    Resize (if too big), convert to gray, denoise, and return grayscale image.
    Resizing keeps aspect ratio and keeps operations fast for large inputs.
    """
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # slight blur to remove small noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def get_skew_angle_tesseract(image: np.ndarray) -> float:
    """Ask Tesseract for orientation and rotate accordingly (returns angle degrees)."""
    try:
        osd = pytesseract.image_to_osd(image)
        # OSD returns lines like "Orientation in degrees: 90"
        m = re.search(r"Rotate:\s*(\d+)", osd)
        if m:
            rot = int(m.group(1))
            # Tesseract Rotate is clockwise degrees that should be rotated to upright
            return -rot  # convert to rotation angle for cv2 (counterclockwise)
    except Exception:
        pass
    return 0.0

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ----------------------
# Table detection - morphological heuristic
# ----------------------
def detect_table_rois_morphology(img_gray: np.ndarray, debug: bool=False) -> List[Tuple[int,int,int,int]]:
    """
    Detect table regions using morphological line detection.
    Returns list of bounding boxes (x, y, w, h).
    Works well for invoices with visible lines or column separators.
    """
    # Binarize
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    # Morph kernels (size tuned by image size)
    h, w = img_gray.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 80)))

    horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    # Combine lines
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # Optional: dilate to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)

    # Find contours on the mask
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = (w*h) * 0.002  # configurable threshold
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww*hh
        # heuristics: exclude tiny boxes and very narrow strips
        if area < min_area or ww < w*0.2 or hh < h*0.02:
            continue
        boxes.append((x,y,ww,hh))

    # sort top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    if debug:
        print(f"[detect_table_rois_morphology] found {len(boxes)} boxes")
    return boxes

# ----------------------
# Optional: layoutparser/Detectron2 based detection
# ----------------------
def detect_table_rois_layoutparser(img_pil: Image.Image) -> List[Tuple[int,int,int,int]]:
    """
    Uses layoutparser to detect tables if available. Returns bounding boxes.
    Requires layoutparser and an available model (Detectron2).
    """
    if not LAYOUTPARSER_AVAILABLE:
        raise RuntimeError("layoutparser not available. Install layoutparser[detectron2] to use this method.")
    # try general object detection model from layoutparser
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                     label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"})
    layout = model.detect(img_pil)
    table_boxes = []
    for bbox in layout:
        if bbox.type.lower() == "table":
            x1,y1,x2,y2 = int(bbox.block.x_1), int(bbox.block.y_1), int(bbox.block.x_2), int(bbox.block.y_2)
            table_boxes.append((x1,y1,x2-x1,y2-y1))
    return table_boxes

# ----------------------
# OCR on ROI
# ----------------------
def ocr_words_from_image(img_bgr: np.ndarray, use_easyocr=False) -> pd.DataFrame:
    """
    Return dataframe of words with bounding boxes and confidence:
    columns: ['text','left','top','width','height','conf','center_x','center_y']
    """
    if use_easyocr and EASYOCR_AVAILABLE:
        reader = easyocr.Reader(['en'], gpu=False)  # languages can be parameterized
        result = reader.readtext(img_bgr[:,:,::-1])  # easyocr expects RGB
        rows = []
        for bbox, text, conf in result:
            # bbox is [top-left, top-right, bottom-right, bottom-left]
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
            rows.append({'text': text, 'left': x, 'top': y, 'width': w, 'height': h, 'conf': conf})
        return pd.DataFrame(rows)
    # pytesseract path
    custom_oem_psm_config = r'--oem 3 --psm 6'  # psm 6 = assume a uniform block of text
    data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DATAFRAME, config=custom_oem_psm_config)
    # pytesseract returns some NaN lines
    data = data.dropna(subset=['text'])
    data = data[data.conf != '-1']
    data = data.rename(columns={'left':'left','top':'top','width':'width','height':'height','conf':'conf','text':'text'})
    # compute centers
    data['center_x'] = data['left'] + data['width'] / 2.0
    data['center_y'] = data['top'] + data['height'] / 2.0
    # convert conf column
    try:
        data['conf'] = data['conf'].astype(float)
    except Exception:
        pass
    return data[['text','left','top','width','height','conf','center_x','center_y']].reset_index(drop=True)

# ----------------------
# Group words into lines/rows
# ----------------------
def group_words_into_rows(ocr_df: pd.DataFrame, y_tol=10) -> List[List[dict]]:
    """
    Cluster words into rows by center_y with tolerance y_tol (in pixels).
    Returns list of rows; each row is list of token dicts with text and x positions.
    Robustness: y_tol can be scaled to the median height of words.
    """
    if ocr_df.empty:
        return []
    # dynamic tolerance based on median text height
    median_h = max(1.0, float(np.median(ocr_df['height'])))
    tol = max(y_tol, median_h * 0.7)
    # sort words by y
    words = ocr_df.sort_values('center_y').to_dict(orient='records')
    rows = []
    current_row = [words[0]]
    current_y = words[0]['center_y']
    for w in words[1:]:
        if abs(w['center_y'] - current_y) <= tol:
            current_row.append(w)
        else:
            # new row
            rows.append(sorted(current_row, key=lambda x: x['center_x']))
            current_row = [w]
            current_y = w['center_y']
    rows.append(sorted(current_row, key=lambda x: x['center_x']))
    return rows

# ----------------------
# Token helpers + detection
# ----------------------
PRICE_RE = re.compile(r'^[\$\£\€\¥\s]*-?[\d\.,]+(?:\.\d{1,4})?$')
INT_RE = re.compile(r'^\d+$')

def normalize_numeric_token(token: str) -> Optional[float]:
    """Strip common currency symbols, parentheses etc., return float or None."""
    if not isinstance(token, str):
        return None
    token = token.strip()
    # remove surrounding parentheses (negatives sometimes)
    token = token.strip("()")
    # remove currency symbols and spaces
    token = re.sub(r'[^0-9\.,\-]', '', token)
    if token == '':
        return None
    # remove thousands separators intelligently:
    # If token has both ',' and '.', decide which is decimal sep
    if token.count(',') > 0 and token.count('.') > 0:
        # assume '.' is decimal if last '.' after last ','
        if token.rfind('.') > token.rfind(','):
            token = token.replace(',', '')
        else:
            token = token.replace('.', '').replace(',', '.')
    else:
        # if only commas and groups of three, remove commas
        if token.count(',') > 0 and re.search(r',\d{3}(?!\d)', token):
            token = token.replace(',', '')
        else:
            token = token.replace(',', '.')
    try:
        val = float(token)
        return val
    except Exception:
        return None

def token_looks_like_price(token: str) -> bool:
    if token is None or token.strip() == '':
        return False
    t = token.strip()
    # common heuristics: contains '.' (cents) or currency symbol or >2 digits
    if re.search(r'[\$\£\€\¥]', t):
        return True
    # digits with decimal
    if re.search(r'\d+\.\d{1,4}$', t):
        return True
    # pure digits but long (>=3) -> maybe price e.g., 1149
    if INT_RE.match(t) and len(t) >= 3:
        return True
    # fallback: price-like numeric
    return bool(re.search(r'^[\d\.,]+$', t))

def token_looks_like_quantity(token: str) -> bool:
    if token is None: return False
    t = token.strip()
    # small integer (1..999) typical
    if INT_RE.match(t):
        try:
            v = int(t)
            if 0 <= v <= 10000:  # generous cap
                # prefer smaller numbers under 500 when considering qty
                return True
        except:
            pass
    return False

# ----------------------
# Parse rows into columns
# ----------------------
HEADER_SYNONYMS = {
    'quantity': ['qty', 'quantity', 'qnty', 'q\'ty', 'q.ty','qty.'],
    'price': ['price', 'unit price', 'unitprice', 'unit_price', 'amount', 'unit', 'unitprice','unitprice( rm )','unit price (rm)'],
    'description': ['description', 'desc', 'item', 'product', 'details']
}

def find_header_row(rows: List[List[dict]]) -> Optional[int]:
    """Search top few rows for header-like tokens and return index if found."""
    top_k = min(6, len(rows))
    for i in range(top_k):
        texts = " ".join([w['text'].lower() for w in rows[i]])
        score = 0
        for k, syns in HEADER_SYNONYMS.items():
            for s in syns:
                if s in texts:
                    score += 1
        if score >= 1:
            return i
    return None

def map_columns_from_header(header_row: List[dict]) -> List[Tuple[str,float]]:
    """
    Map detected header tokens (with center_x) to column semantic names.
    Returns list of (semantic_label, x_center) sorted by x.
    """
    mapping = []
    for token in header_row:
        t = token['text'].lower().strip()
        # find best match among synonyms
        found = None
        for label, syns in HEADER_SYNONYMS.items():
            for s in syns:
                if s in t:
                    found = label
                    break
            if found:
                break
        if found:
            mapping.append((found, token['center_x']))
    # sort by x and return
    mapping = sorted(mapping, key=lambda x: x[1])
    return mapping

def assign_tokens_to_columns_by_x(row_tokens: List[dict], col_x_positions: List[float]) -> dict:
    """
    Given a row (tokens) and column x centers, assign tokens to nearest column (by center_x).
    Returns dict col_index -> joined text.
    """
    assignments = {i: [] for i in range(len(col_x_positions))}
    for t in row_tokens:
        x = t['center_x']
        # find nearest column center
        deltas = [abs(x - cx) for cx in col_x_positions]
        idx = int(np.argmin(deltas))
        assignments[idx].append(t['text'])
    # join
    assignments = {i: " ".join(v).strip() for i,v in assignments.items()}
    return assignments

def parse_table_rows(rows: List[List[dict]]) -> pd.DataFrame:
    """
    Core heuristic to extract (quantity, price, description) from grouped rows.
    Uses header mapping if present, otherwise per-row heuristics.
    """
    parsed = []
    header_idx = find_header_row(rows)
    col_centers = None
    semantic_map = None
    if header_idx is not None:
        header = rows[header_idx]
        mapping = map_columns_from_header(header)
        if mapping:
            # build column x centers in document order
            semantic_map = [label for label,_ in mapping]
            col_centers = [x for _,x in mapping]

    # iterate rows skipping header row
    for i, row in enumerate(rows):
        if i == header_idx:
            continue
        # prepare tokens text list
        tokens = [t['text'] for t in row]
        texts = [t['text'] for t in row]
        # case 1: header mapping available -> assign tokens by x
        if col_centers is not None and len(col_centers) >= 2:
            assigned = assign_tokens_to_columns_by_x(row, col_centers)
            # find column indices for qty/price/desc based on semantic_map
            # if semantic_map contains the three types, map them directly
            desc = ""
            qty = None
            price = None
            for idx,label in enumerate(semantic_map):
                txt = assigned.get(idx, "")
                if label == 'description':
                    desc = txt
                elif label == 'quantity':
                    qty = txt
                elif label == 'price':
                    price = txt
            # fallback: if mapping not complete, fall back to heuristics
            if desc.strip() != "" or qty or price:
                parsed.append({'raw_qty': qty, 'raw_price': price, 'raw_description': desc})
                continue

        # case 2: no header mapping — heuristics per row
        # find candidate price tokens (prefer rightmost tokens that look like price)
        price_token = None
        quantity_token = None
        # search right-to-left for price-looking token
        for tok in reversed(texts):
            if token_looks_like_price(tok):
                price_token = tok
                break
        # search left-to-right for small-int token for qty
        for tok in texts:
            if token_looks_like_quantity(tok):
                # avoid taking the price token as qty
                if price_token and tok == price_token:
                    continue
                quantity_token = tok
                break
        # Description: everything except chosen qty & price (join with spaces)
        desc_tokens = []
        for tok in texts:
            if tok == price_token and price_token is not None:
                continue
            if tok == quantity_token and quantity_token is not None:
                continue
            desc_tokens.append(tok)
        description = " ".join(desc_tokens).strip()
        parsed.append({'raw_qty': quantity_token, 'raw_price': price_token, 'raw_description': description})

    df = pd.DataFrame(parsed)
    # cleaning
    df['quantity'] = df['raw_qty'].apply(lambda x: None if x is None else (int(normalize_numeric_token(x)) if normalize_numeric_token(x) is not None and float(normalize_numeric_token(x)).is_integer() else normalize_numeric_token(x)))
    df['price'] = df['raw_price'].apply(lambda x: None if x is None else normalize_numeric_token(x))
    df['description'] = df['raw_description'].astype(str).apply(lambda s: re.sub(r'\s+', ' ', s).strip())
    # final columns
    return df[['quantity','price','description']]

# ----------------------
# High-level orchestrator
# ----------------------
def extract_table_from_invoice(path_or_pil, prefer_layoutparser=False, use_easyocr=False, debug=False) -> pd.DataFrame:
    """
    Main entrypoint. Returns a DataFrame with columns: quantity, price, description
    """
    img_bgr = read_image(path_or_pil)  # BGR
    # preprocess (grayscale & resize)
    gray = preprocess_for_table(img_bgr)

    # deskew whole page a bit
    angle = get_skew_angle_tesseract(gray)
    if debug:
        print(f"[extract] page skew angle (deg): {angle}")
    if abs(angle) > 0.1:
        img_bgr = rotate_image(img_bgr, angle)
        gray = preprocess_for_table(img_bgr)

    # detect tables with morphology
    boxes = detect_table_rois_morphology(gray, debug=debug)

    # If none found and user prefers ML or morphology failed, try layoutparser
    if (len(boxes) == 0 and prefer_layoutparser) or (len(boxes) == 0 and LAYOUTPARSER_AVAILABLE):
        try:
            pil = cv2_to_pil(img_bgr)
            lp_boxes = detect_table_rois_layoutparser(pil)
            if lp_boxes:
                boxes = lp_boxes
        except Exception as e:
            if debug:
                print("[extract] layoutparser detection failed:", e)

    # If still none found => fallback: use full page as single table ROI
    if not boxes:
        h,w = img_bgr.shape[:2]
        boxes = [(0,0,w,h)]
        if debug:
            print("[extract] no boxes detected, fallback to full page.")

    # For each ROI, OCR and parse; accumulate rows across ROIs
    all_parsed = []
    for box in boxes:
        x,y,w,h = box
        roi = img_bgr[y:y+h, x:x+w]
        # optional further deskew/threshold inside ROI could be applied
        ocr_df = ocr_words_from_image(roi, use_easyocr=use_easyocr)
        if ocr_df.empty:
            continue
        rows = group_words_into_rows(ocr_df)
        parsed_df = parse_table_rows(rows)
        # drop empty rows
        parsed_df = parsed_df[parsed_df['description'].str.strip() != ""].reset_index(drop=True)
        all_parsed.append(parsed_df)

    if not all_parsed:
        # final fallback: OCR entire page as plain text and attempt heuristic extraction by regex (qty/price)
        if debug:
            print("[extract] No table ROIs yielded parsed rows. Final fallback: OCR full page text.")
        ocr_full = ocr_words_from_image(img_bgr, use_easyocr=use_easyocr)
        rows = group_words_into_rows(ocr_full)
        parsed_df = parse_table_rows(rows)
        return parsed_df

    df_final = pd.concat(all_parsed, ignore_index=True)
    # final normalization: drop duplicates and fill nulls with heuristics
    df_final = df_final.drop_duplicates().reset_index(drop=True)

    # If quantity missing but price present, sometimes quantity is '5' repeated per item row; leave None if not found
    return df_final

# ----------------------
# Example usage:
# ----------------------
if __name__ == "__main__":
    # quick test with one of the images shipped in container (change path accordingly)
    sample_path = "PclHttp_2_4[1]_page_1.png"
    if os.path.exists(sample_path):
        df = extract_table_from_invoice(sample_path, prefer_layoutparser=False, use_easyocr=False, debug=True)
        print(df.head(30))
        df.to_csv("extracted_table.csv", index=False)
    else:
        print("Change sample_path to an invoice image or PDF file to run this example.")
