# extract_deck2_colors.py
"""
Extract only blue and yellow cards from Deck 2 PDF.
Deck 2 layout: Skip first 20 cards (violet/red duplicates), then Blue 0-9, Yellow 0-9
"""

from pathlib import Path
from pdf2image import convert_from_path

ASSETS_DIR = Path(__file__).parent / "assets"
PDF_PATH = ASSETS_DIR / "PDF" / "DD_GF_Code_Deck2_Front.pdf"
OUTPUT_DIR = ASSETS_DIR / "cards" / "number"
POPPLER_PATH = r"C:\poppler\poppler-24.02.0\Library\bin"
DPI = 200

def extract_blue_yellow():
    print("Extracting blue and yellow cards from Deck 2...")
    print(f"PDF: {PDF_PATH}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert PDF pages to images
    print("Converting PDF (this may take a minute)...")
    pages = convert_from_path(str(PDF_PATH), dpi=DPI, poppler_path=POPPLER_PATH)
    print(f"Found {len(pages)} pages")
    
    # Blue cards: pages 20-29 (index 20-29)
    print("\nExtracting Blue 0-9...")
    for i in range(10):
        page_idx = 20 + i
        if page_idx < len(pages):
            for copy in [1, 2]:
                filename = f"blau_{i}_copy{copy}.png"
                filepath = OUTPUT_DIR / filename
                pages[page_idx].save(filepath, "PNG")
            print(f"  Saved blau_{i}")
    
    # Yellow cards: pages 30-39 (index 30-39)
    print("\nExtracting Yellow 0-9...")
    for i in range(10):
        page_idx = 30 + i
        if page_idx < len(pages):
            for copy in [1, 2]:
                filename = f"gelb_{i}_copy{copy}.png"
                filepath = OUTPUT_DIR / filename
                pages[page_idx].save(filepath, "PNG")
            print(f"  Saved gelb_{i}")
    
    print("\nDone!")

if __name__ == "__main__":
    extract_blue_yellow()
