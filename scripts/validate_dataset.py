#!/usr/bin/env python3
"""Validate Fashionpedia dataset after download."""
import json
import sys
from pathlib import Path

from PIL import Image


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images" / "fashionpedia"


def validate_annotations():
    """Check annotation files exist and are valid JSON."""
    print("Validating annotations...")

    ann_file = ANNOTATIONS_DIR / "val_annotations.json"
    if not ann_file.exists():
        print(f"✗ Missing: {ann_file}")
        return False

    try:
        with open(ann_file) as f:
            data = json.load(f)

        num_images = len(data.get("images", []))
        num_annotations = len(data.get("annotations", []))
        num_categories = len(data.get("categories", []))

        print(f"✓ Annotations valid:")
        print(f"  - {num_images} images")
        print(f"  - {num_annotations} annotations")
        print(f"  - {num_categories} categories")

        if num_images == 0 or num_annotations == 0:
            print("✗ Annotation file appears empty")
            return False

        return True
    except Exception as e:
        print(f"✗ Invalid annotation file: {e}")
        return False


def validate_images():
    """Check image directory and sample a few images."""
    print("\nValidating images...")

    if not IMAGES_DIR.exists():
        print(f"✗ Images directory not found: {IMAGES_DIR}")
        return False

    image_files = list(IMAGES_DIR.glob("*.jpg"))
    num_images = len(image_files)

    if num_images == 0:
        print(f"✗ No images found in {IMAGES_DIR}")
        return False

    print(f"✓ Found {num_images} images")

    # Sample 3 images
    print("  Checking 3 sample images...")
    for img_path in image_files[:3]:
        try:
            with Image.open(img_path) as img:
                img.verify()
            print(f"    ✓ {img_path.name}")
        except Exception as e:
            print(f"    ✗ {img_path.name}: {e}")
            return False

    return True


def validate_image_annotation_match():
    """Check that annotated images exist."""
    print("\nValidating image-annotation correspondence...")

    ann_file = ANNOTATIONS_DIR / "val_annotations.json"
    with open(ann_file) as f:
        data = json.load(f)

    missing = 0
    for img_info in data["images"][:10]:  # Check first 10
        img_name = img_info["file_name"]
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"  ✗ Annotated image missing: {img_name}")
            missing += 1

    if missing > 0:
        print(f"✗ {missing}/10 sampled images missing")
        return False

    print("✓ Image-annotation correspondence OK (sampled 10)")
    return True


def main():
    print("=" * 60)
    print("Fashionpedia Dataset Validator")
    print("=" * 60)
    print()

    checks = [
        validate_annotations(),
        validate_images(),
        validate_image_annotation_match(),
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All validation checks passed!")
        print("Dataset is ready for use.")
    else:
        print("✗ Validation failed!")
        print("Run scripts/download_fashionpedia.py to fix.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
