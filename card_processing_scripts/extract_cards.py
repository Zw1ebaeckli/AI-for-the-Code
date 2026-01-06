# extract_cards.py
"""
Extract individual card images from the CODE game PDF files.
Each page in the PDF is a single card.

Requirements:
    pip install pdf2image Pillow

Note: pdf2image requires poppler to be installed:
    - Windows: Download from https://github.com/osber/poppler/releases 
               or install via: choco install poppler
    - Mac: brew install poppler
    - Linux: apt-get install poppler-utils
"""

import os
from pathlib import Path

# Check if required packages are installed
try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Required packages not installed. Please run:")
    print("  pip install pdf2image Pillow")
    print("\nAlso install poppler:")
    print("  Windows: choco install poppler (or download manually)")
    print("  Mac: brew install poppler")
    print("  Linux: apt-get install poppler-utils")
    exit(1)

# Configuration
ASSETS_DIR = Path(__file__).parent / "assets"
PDF_DIR = ASSETS_DIR / "PDF"
OUTPUT_DIR = ASSETS_DIR / "cards"

# Poppler path for Windows (adjust if installed elsewhere)
POPPLER_PATH = r"C:\poppler\poppler-24.02.0\Library\bin"

# PDF files
PDF_FILES = {
    "deck1_front": "DD_GF_Code_Deck1_Front_02.pdf",
    "deck1_back": "DD_GF_Code_Deck1_Back.pdf",
    "deck2_front": "DD_GF_Code_Deck2_Front.pdf",
    "deck2_back": "DD_GF_Code_Deck2_Back.pdf",
}

DPI = 200  # Resolution for PDF conversion (200 is good balance of quality/size)

# The official 10 code cards (first 10 pages of deck 1)
OFFICIAL_CODES = [
    "3045", "2489", "2357", "1479", "1368",
    "1258", "0459", "0278", "0169", "3456",
]

# Card layout in Deck 1 (after the 10 code cards, starting at index 10):
# Index 10-19: Purple 0-9
# Index 20-29: Red 0-9
# Index 30-34: AUSSETZEN (5 cards)
# Index 35-38: JOKER (4 cards)
# Index 39-44: TAUSCH (6 cards)
# Index 45-48: PLUS2 (4 cards)
# Index 49-52: GESCHENK (4 cards)
# Index 53: RESET (1 card)
# Index 54-58: RICHTUNGSWECHSEL (5 cards)

# Card layout in Deck 2:
# Index 0-19: SKIP (duplicates of purple/red from deck 1)
# Index 20-29: Blue 0-9
# Index 30-39: Yellow 0-9

DECK1_LAYOUT = [
    # (start_idx, count, color_or_action, card_type)
    (10, 10, "VIOLETT", "NUMBER"),   # Purple 0-9
    (20, 10, "ROT", "NUMBER"),       # Red 0-9
    (30, 5, "AUSSETZEN", "ACTION"),
    (35, 4, "JOKER", "ACTION"),
    (39, 6, "TAUSCH", "ACTION"),
    (45, 4, "PLUS2", "ACTION"),
    (49, 4, "GESCHENK", "ACTION"),
    (53, 1, "RESET", "ACTION"),
    (54, 5, "RICHTUNGSWECHSEL", "ACTION"),
]

DECK2_LAYOUT = [
    # Skip first 20 cards (index 0-19) - duplicates of violet/red
    (20, 10, "BLAU", "NUMBER"),      # Blue 0-9
    (30, 10, "GELB", "NUMBER"),      # Yellow 0-9
]


def extract_all_cards():
    """Extract all cards from PDFs - each page is one card."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (OUTPUT_DIR / "code").mkdir(exist_ok=True)
    (OUTPUT_DIR / "number").mkdir(exist_ok=True)
    (OUTPUT_DIR / "action").mkdir(exist_ok=True)
    (OUTPUT_DIR / "back").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CODE Card Extractor")
    print("=" * 60)
    
    extracted = {"code": [], "number": [], "action": [], "backs": []}
    
    for name, filename in PDF_FILES.items():
        pdf_path = PDF_DIR / filename
        
        if not pdf_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"\nProcessing {filename}...")
        
        try:
            pages = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=POPPLER_PATH)
            print(f"  Found {len(pages)} page(s)")
            
            is_back = "back" in name.lower()
            deck = "deck1" if "deck1" in name else "deck2"
            
            if is_back:
                # Save one back card per deck
                card_filename = f"{deck}_back.png"
                card_path = OUTPUT_DIR / "back" / card_filename
                pages[0].save(card_path, "PNG")
                extracted["backs"].append(card_path)
                print(f"  Saved: back/{card_filename}")
                continue
            
            # Process front cards based on deck
            if deck == "deck1":
                # First 10 are code cards
                for i in range(min(10, len(pages))):
                    code_value = OFFICIAL_CODES[i]
                    card_filename = f"code_{code_value}.png"
                    card_path = OUTPUT_DIR / "code" / card_filename
                    pages[i].save(card_path, "PNG")
                    extracted["code"].append(card_path)
                print(f"  Saved 10 code cards")
                
                # Process rest according to layout
                for start_idx, count, name_part, card_type in DECK1_LAYOUT:
                    for j in range(count):
                        page_idx = start_idx + j
                        if page_idx >= len(pages):
                            print(f"  Warning: Page {page_idx} not found for {name_part}")
                            continue
                        
                        if card_type == "NUMBER":
                            # Number cards: save twice (game has 2 copies each)
                            value = j  # 0-9
                            for copy in range(1, 3):
                                card_filename = f"{name_part.lower()}_{value}_copy{copy}.png"
                                card_path = OUTPUT_DIR / "number" / card_filename
                                pages[page_idx].save(card_path, "PNG")
                                extracted["number"].append(card_path)
                        else:
                            # Action cards
                            card_filename = f"{name_part.lower()}_{j+1}.png"
                            card_path = OUTPUT_DIR / "action" / card_filename
                            pages[page_idx].save(card_path, "PNG")
                            extracted["action"].append(card_path)
                
                print(f"  Saved number and action cards from deck 1")
            
            else:  # deck2
                # Skip first 18 cards, process blue and yellow
                for start_idx, count, color, card_type in DECK2_LAYOUT:
                    for j in range(count):
                        page_idx = start_idx + j
                        if page_idx >= len(pages):
                            print(f"  Warning: Page {page_idx} not found for {color}")
                            continue
                        
                        # Number cards: save twice
                        value = j  # 0-9
                        for copy in range(1, 3):
                            card_filename = f"{color.lower()}_{value}_copy{copy}.png"
                            card_path = OUTPUT_DIR / "number" / card_filename
                            pages[page_idx].save(card_path, "PNG")
                            extracted["number"].append(card_path)
                
                print(f"  Saved blue and yellow number cards from deck 2")
                
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"  Code cards: {len(extracted['code'])}")
    print(f"  Number cards: {len(extracted['number'])} (includes duplicates)")
    print(f"  Action cards: {len(extracted['action'])}")
    print(f"  Back cards: {len(extracted['backs'])}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print card summary
    print("\nCard files created:")
    print("  code/code_XXXX.png - 10 secret code cards")
    print("  number/{color}_{0-9}_copy{1-2}.png - 80 number cards (4 colors × 10 values × 2 copies)")
    print("  action/{type}_{n}.png - 30 action cards")
    print("  back/deck{1,2}_back.png - card back images")
    
    return extracted


def list_extracted_cards():
    """List all extracted card images."""
    if not OUTPUT_DIR.exists():
        print("No cards extracted yet. Run extract_all_cards() first.")
        return
    
    cards = list(OUTPUT_DIR.glob("*.png"))
    print(f"Found {len(cards)} extracted cards:")
    for card in sorted(cards):
        print(f"  {card.name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_extracted_cards()
    else:
        extract_all_cards()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        # Full extraction with custom parameters
        rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        cols = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        extract_cards_final(cards_per_row=cols, cards_per_col=rows)
    else:
        # Preview mode to determine layout
        extract_all_cards()
