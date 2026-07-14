#!/usr/bin/env python3
"""
Download Fashionpedia images using the annotations file.

Fashionpedia images are hosted on various URLs. This script:
1. Reads the annotations to get image URLs
2. Downloads images to data/raw/images/fashionpedia/
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import requests
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "val_annotations.json"
IMAGES_DIR = DATA_DIR / "images" / "fashionpedia"


def download_image(img_info, output_dir):
    """Download a single image."""
    filename = img_info['file_name']
    url = img_info['original_url']
    output_path = output_dir / filename

    if output_path.exists():
        return True, filename

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return True, filename
    except Exception as e:
        return False, f"{filename}: {e}"


def main():
    print("=" * 60)
    print("Fashionpedia Image Downloader")
    print("=" * 60)

    print("\nLoading annotations...")
    with open(ANNOTATIONS_FILE) as f:
        data = json.load(f)

    images = data['images']
    print(f"Found {len(images)} images to download")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(IMAGES_DIR.glob("*.jpg")))
    if existing > 0:
        print(f"{existing} images already downloaded")

    print("\nDownloading images...")
    print("This will take a while (~1000 images, downloading from various sources)")

    successful = 0
    failed = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_image, img, IMAGES_DIR): img for img in images}

        with tqdm(total=len(images)) as pbar:
            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    successful += 1
                else:
                    failed.append(result)
                pbar.update(1)

    print(f"\n" + "=" * 60)
    print(f"Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(failed)}")
    if failed and len(failed) <= 10:
        print(f"\nFailed downloads:")
        for f in failed[:10]:
            print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
