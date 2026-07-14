#!/usr/bin/env python3
"""
Download Fashionpedia dataset (validation split).

Fashionpedia is hosted at https://fashionpedia.github.io/home/index.html
Annotations follow COCO format and are available via the official links.
"""
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import requests
from tqdm import tqdm


FASHIONPEDIA_URLS = {
    "val_annotations": "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
    "val_images_info": "https://s3.amazonaws.com/ifashionist-dataset/annotations/info_val2020.json",
}

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"


def download_with_progress(url: str, dest: Path):
    """Download file with progress bar."""
    print(f"Downloading {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_annotations():
    """Download Fashionpedia annotations."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in FASHIONPEDIA_URLS.items():
        dest = ANNOTATIONS_DIR / f"{name}.json"
        if dest.exists():
            print(f"✓ {dest.name} already exists")
            continue

        try:
            download_with_progress(url, dest)
          
            with open(dest) as f:
                json.load(f)
            print(f"✓ {dest.name} downloaded and validated")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            if dest.exists():
                dest.unlink()
            return False

    return True


def download_images():
    """Download COCO validation images (used by Fashionpedia)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "val2017.zip"


    if (IMAGES_DIR / "val2017").exists():
        num_images = len(list((IMAGES_DIR / "val2017").glob("*.jpg")))
        if num_images > 1000:
            print(f"✓ Images already exist ({num_images} files)")
            return True


    if not zip_path.exists():
        print("\nDownloading COCO validation images (~1GB, this may take a while)...")
        print("Alternative: Download manually from http://images.cocodataset.org/zips/val2017.zip")
        print("and place in data/raw/, then re-run this script.\n")

        try:
            download_with_progress(COCO_VAL_URL, zip_path)
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            print("\nMANUAL DOWNLOAD REQUIRED:")
            print(f"1. Download: {COCO_VAL_URL}")
            print(f"2. Place at: {zip_path}")
            print("3. Re-run this script")
            return False


    print("Extracting images...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(IMAGES_DIR)
        print(f"✓ Extracted to {IMAGES_DIR / 'val2017'}")

  
        zip_path.unlink()
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Fashionpedia Dataset Downloader")
    print("=" * 60)


    print("\n[1/2] Downloading annotations...")
    if not download_annotations():
        sys.exit(1)

    print("\n[2/2] Downloading images...")
    if not download_images():
        print("\n⚠ Images not downloaded automatically.")
        print("Continue with manual download or retry later.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Dataset download complete!")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Images: {IMAGES_DIR / 'val2017'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
