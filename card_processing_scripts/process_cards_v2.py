# process_cards_v2.py
"""
Post-process extracted card images with custom parameters.

Usage:
    Preview mode (test on one card):
        python process_cards_v2.py --preview [card_path] [crop_pixels] [corner_radius]
        Example: python process_cards_v2.py --preview assets/cards/code/code_1479.png 25 0.08
    
    Process all cards:
        python process_cards_v2.py [crop_pixels] [corner_radius]
        Example: python process_cards_v2.py 25 0.08
    
    Parameters:
        crop_pixels: Number of pixels to remove from all sides (default: 25)
        corner_radius: Corner radius as fraction of width, e.g., 0.08 = 8% (default: 0.08)
"""

from pathlib import Path
from PIL import Image, ImageDraw
import sys

ASSETS_DIR = Path(__file__).parent / "assets"
CARDS_DIR = ASSETS_DIR / "cards"
OUTPUT_DIR = ASSETS_DIR / "cards_processed"
TEST_DIR = ASSETS_DIR / "cards_test"  # For preview images


def crop_card(img: Image.Image, crop_pixels: int) -> Image.Image:
    """
    Crop the image by removing crop_pixels from all sides.
    """
    left = crop_pixels
    top = crop_pixels
    right = img.width - crop_pixels
    bottom = img.height - crop_pixels
    
    return img.crop((left, top, right, bottom))


def add_rounded_corners(img: Image.Image, corner_radius_percent: float) -> Image.Image:
    """
    Add rounded corners with transparency.
    corner_radius_percent: radius as fraction of card width (e.g., 0.08 = 8%)
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    radius = int(img.width * corner_radius_percent)
    
    # Create mask
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [(0, 0), (img.width - 1, img.height - 1)],
        radius=radius,
        fill=255
    )
    
    # Apply mask
    output = Image.new('RGBA', img.size, (0, 0, 0, 0))
    output.paste(img, mask=mask)
    
    return output


def process_card(input_path: Path, output_path: Path, crop_pixels: int, corner_radius_percent: float) -> bool:
    """Process a single card image."""
    try:
        img = Image.open(input_path)
        
        # Crop from all sides
        cropped = crop_card(img, crop_pixels)
        
        # Add rounded corners
        result = add_rounded_corners(cropped, corner_radius_percent)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, 'PNG')
        
        return True
    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")
        return False


def process_all_cards(crop_pixels: int, corner_radius_percent: float):
    """Process all extracted card images, overwriting originals."""
    print("=" * 70)
    print("Card Image Processor - OVERWRITING ORIGINALS")
    print(f"Settings:")
    print(f"  - Crop: {crop_pixels}px from all sides")
    print(f"  - Corner radius: {corner_radius_percent*100:.1f}% of card width")
    print("=" * 70)
    
    total = 0
    success = 0
    
    # Process each subdirectory (except back, handled separately)
    for subdir in ['code', 'number', 'action']:
        input_dir = CARDS_DIR / subdir
        
        if not input_dir.exists():
            print(f"\nSkipping {subdir}/ (not found)")
            continue
        
        cards = [c for c in input_dir.glob('*.png') if not c.name.startswith('preview')]
        print(f"\nProcessing {subdir}/ ({len(cards)} cards)...")
        
        for card_path in cards:
            # Overwrite the original file
            if process_card(card_path, card_path, crop_pixels, corner_radius_percent):
                success += 1
            total += 1
    
    # Process back cards
    print("\nProcessing back cards...")
    back_dir = CARDS_DIR / "back"
    
    deck1_back = back_dir / "deck1_back.png"
    deck2_back = back_dir / "deck2_back.png"
    
    if deck1_back.exists():
        if process_card(deck1_back, deck1_back, crop_pixels, corner_radius_percent):
            success += 1
            print("  Processed: deck1_back.png")
        total += 1
    
    if deck2_back.exists():
        if process_card(deck2_back, deck2_back, crop_pixels, corner_radius_percent):
            success += 1
            print("  Processed: deck2_back.png")
        total += 1
    
    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"  Processed: {success}/{total} cards")
    print(f"  Cards updated in: {CARDS_DIR}")
    print("=" * 70)


def preview_card(card_path: Path, crop_pixels: int, corner_radius_percent: float):
    """Preview processing on a single card."""
    if not card_path.exists():
        print(f"Card not found: {card_path}")
        return
    
    # Create test directory
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Preview Mode - Testing on: {card_path.name}")
    print(f"Settings:")
    print(f"  - Crop: {crop_pixels}px from all sides")
    print(f"  - Corner radius: {corner_radius_percent*100:.1f}% of card width")
    print("=" * 70)
    
    img = Image.open(card_path)
    
    print(f"\nOriginal size: {img.width} x {img.height}")
    
    # Crop
    cropped = crop_card(img, crop_pixels)
    print(f"After crop: {cropped.width} x {cropped.height}")
    
    # Save preview with crop bounds drawn
    preview = img.convert('RGB').copy()
    draw = ImageDraw.Draw(preview)
    left, top = crop_pixels, crop_pixels
    right, bottom = img.width - crop_pixels, img.height - crop_pixels
    draw.rectangle([(left, top), (right, bottom)], outline='red', width=3)
    
    preview_path = TEST_DIR / f"preview_bounds_{card_path.name}"
    preview.save(preview_path)
    print(f"\nCrop bounds preview: {preview_path}")
    
    # Save final processed preview
    result = add_rounded_corners(cropped, corner_radius_percent)
    
    rounded_path = TEST_DIR / f"preview_final_{card_path.name}"
    result.save(rounded_path, 'PNG')
    print(f"Final preview:       {rounded_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Parse command line arguments
    crop_pixels = 25  # default
    corner_radius = 0.08  # default
    
    if "--preview" in sys.argv:
        # Preview mode
        preview_idx = sys.argv.index("--preview")
        
        # Get card path (default to code_1479.png)
        if len(sys.argv) > preview_idx + 1 and not sys.argv[preview_idx + 1].replace('.', '').replace('-', '').isdigit():
            card_path = Path(sys.argv[preview_idx + 1])
        else:
            card_path = CARDS_DIR / "code" / "code_1479.png"
        
        # Get crop pixels
        if len(sys.argv) > preview_idx + 2:
            try:
                crop_pixels = int(sys.argv[preview_idx + 2])
            except ValueError:
                pass
        
        # Get corner radius
        if len(sys.argv) > preview_idx + 3:
            try:
                corner_radius = float(sys.argv[preview_idx + 3])
            except ValueError:
                pass
        
        preview_card(card_path, crop_pixels, corner_radius)
    else:
        # Process all mode
        if len(sys.argv) > 1:
            try:
                crop_pixels = int(sys.argv[1])
            except ValueError:
                pass
        
        if len(sys.argv) > 2:
            try:
                corner_radius = float(sys.argv[2])
            except ValueError:
                pass
        
        process_all_cards(crop_pixels, corner_radius)
