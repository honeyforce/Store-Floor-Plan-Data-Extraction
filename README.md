# Clip-Strip Extractor (UI)

A simple Streamlit app to:
- Detect highlighted bay labels in a floorplan image
- OCR the location codes (e.g., `7-R-18`)
- Look up `Adjacency` and `Preferred / 2nd / 3rd` from your Excel
- Export results as an Excel file

## Setup

1. Python 3.10+ recommended
2. Install Tesseract OCR:
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
3. Install Python packages:
   ```bash
   pip install -r requirements.txt
