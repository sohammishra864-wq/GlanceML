import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.colour_palette import get_nearest_color_name, normalize_color_name


def extract_dominant_color(image: Image.Image, mask: np.ndarray, num_clusters: int = 5) -> str:
    # Convert image to numpy array
    img_array = np.array(image)

    # Get only the pixels where the mask is 1 (the garment area)
    garment_pixels = img_array[mask.astype(bool)]

    if len(garment_pixels) == 0:
        return "unknown"

    # Cluster the pixels to find dominant colors
    # Use at most 5 clusters, or fewer if we don't have enough pixels
    k = min(num_clusters, len(garment_pixels))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(garment_pixels)

    # Find which cluster has the most pixels
    cluster_labels = kmeans.labels_
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    biggest_cluster = unique_labels[np.argmax(counts)]

    # Get the average RGB color of that cluster
    dominant_rgb = kmeans.cluster_centers_[biggest_cluster]
    dominant_rgb = tuple(dominant_rgb.astype(int))

    # Match to nearest named color
    color_name = get_nearest_color_name(dominant_rgb)

    # Normalize synonyms (e.g., navy -> blue)
    return normalize_color_name(color_name)


def get_garment_colors(image: Image.Image, annotations: list, ann_size: tuple = None) -> list[tuple[str, str]]:
    garment_color_pairs = []
    img_w, img_h = image.size
    # Scale factor when annotations reference a different resolution than the image on disk
    sx = sy = 1.0
    if ann_size:
        sx = img_w / ann_size[0]
        sy = img_h / ann_size[1]

    for ann in annotations:
        garment_type = ann.get('category_name', 'unknown')

        if 'segmentation' in ann and ann['segmentation']:
            mask = create_mask_from_segmentation(ann['segmentation'], image.size, sx, sy)
        elif 'bbox' in ann:
            mask = create_mask_from_bbox(ann['bbox'], image.size, sx, sy)
        else:
            continue

        # Extract the dominant color
        color = extract_dominant_color(image, mask)

        if color != "unknown":
            garment_color_pairs.append((garment_type, color))

    return garment_color_pairs


def create_mask_from_bbox(bbox: list, image_size: tuple, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
    """
    Create a binary mask from a bounding box.
    bbox format: [x, y, width, height]
    """
    x, y, w, h = bbox
    x, w = x * sx, w * sx
    y, h = y * sy, h * sy
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    mask[int(y):int(y+h), int(x):int(x+w)] = 1
    return mask


def create_mask_from_segmentation(seg, image_size: tuple, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
    from PIL import ImageDraw

    mask = Image.new('L', image_size, 0)

    if isinstance(seg, list) and len(seg) > 0:
        draw = ImageDraw.Draw(mask)
        for polygon in seg:
            if len(polygon) >= 6:
                points = [(polygon[i] * sx, polygon[i+1] * sy) for i in range(0, len(polygon), 2)]
                draw.polygon(points, outline=1, fill=1)

    return np.array(mask)


if __name__ == "__main__":
    # Simple test
    print("Testing color extraction...")

    # Test with a red image
    test_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    test_mask = np.ones((100, 100), dtype=np.uint8)

    color = extract_dominant_color(test_img, test_mask, num_clusters=1)
    print(f"Red image -> '{color}'")
    assert color == "red"

    # Test with a blue image
    test_img = Image.new('RGB', (100, 100), color=(0, 0, 200))
    color = extract_dominant_color(test_img, test_mask, num_clusters=1)
    print(f"Blue image -> '{color}'")
    assert color == "blue"

    print("Color extraction works!")
