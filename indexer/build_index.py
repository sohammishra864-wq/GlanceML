import json
import sqlite3
import sys
from pathlib import Path
from tqdm import tqdm

import faiss
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.color_extractor import get_garment_colors
from indexer.scene_classifier import SceneClassifier
from indexer.style_classifier import StyleClassifier
from models.backbone import BackboneModel


# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "val_annotations.json"
IMAGES_DIR = DATA_DIR / "images" / "fashionpedia"  # Fashionpedia has its own image set
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

FAISS_INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
METADATA_DB_FILE = OUTPUT_DIR / "metadata.db"


def load_dataset():
    print("Loading Fashionpedia dataset...")

    with open(ANNOTATIONS_FILE) as f:
        data = json.load(f)

    # Create lookups
    images_by_id = {img['id']: img for img in data['images']}
    categories_by_id = {cat['id']: cat['name'] for cat in data['categories']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []

        # Add category name
        ann['category_name'] = categories_by_id.get(ann['category_id'], 'unknown')
        annotations_by_image[img_id].append(ann)

    print(f"Loaded {len(images_by_id)} images")
    print(f"{len(annotations_by_image)} images have garment annotations")

    return images_by_id, annotations_by_image


def create_metadata_database():
    """Create SQLite database for image metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old database if exists
    if METADATA_DB_FILE.exists():
        METADATA_DB_FILE.unlink()

    conn = sqlite3.connect(METADATA_DB_FILE)
    cursor = conn.cursor()

    # Create table
    # pairs: JSON string like '["shirt:white", "tie:red"]' (BOUND TOKENS - critical!)
    cursor.execute('''
        CREATE TABLE image_metadata (
            image_id TEXT PRIMARY KEY,
            filename TEXT,
            pairs TEXT,
            scene TEXT,
            style TEXT
        )
    ''')

    conn.commit()
    return conn


def extract_features(image, annotations, scene_classifier, style_classifier, ann_size=None):
    # Extract garment-color pairs (IMPORTANT: keeps binding intact)
    garment_color_pairs = get_garment_colors(image, annotations, ann_size=ann_size)

    # Classify scene
    scene = scene_classifier.classify(image)

    # Classify style (based on garment types)
    garment_types = [g for g, c in garment_color_pairs]
    style = style_classifier.classify(garment_types)

    # Format pairs as bound tokens: "garment:color"
    pairs_list = [f"{garment}:{color}" for garment, color in garment_color_pairs]

    return {
        'pairs': pairs_list,
        'scene': scene,
        'style': style
    }


def build_index(images_by_id, annotations_by_image, batch_size=32):
    print("\n" + "="*60)
    print("Building Index")
    print("="*60)

    # Initialize models
    print("\nLoading models...")
    backbone = BackboneModel()
    scene_classifier = SceneClassifier(model=backbone)  # Reuse same model
    style_classifier = StyleClassifier()

    # Create metadata database
    print("\nCreating metadata database...")
    db_conn = create_metadata_database()
    cursor = db_conn.cursor()

    # Process images
    print(f"\nProcessing {len(images_by_id)} images...")

    all_embeddings = []
    all_image_ids = []
    processed_count = 0
    skipped_count = 0

    for img_id, img_info in tqdm(images_by_id.items(), desc="Indexing"):
        filename = img_info['file_name']
        img_path = IMAGES_DIR / filename

        # Skip if image doesn't exist
        if not img_path.exists():
            skipped_count += 1
            continue

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Generate embedding
            embedding = backbone.embed_image(image)
            all_embeddings.append(embedding)
            all_image_ids.append(str(img_id))

            # Extract features (only if we have annotations)
            if img_id in annotations_by_image:
                ann_size = (img_info.get('width', image.size[0]), img_info.get('height', image.size[1]))
                features = extract_features(
                    image,
                    annotations_by_image[img_id],
                    scene_classifier,
                    style_classifier,
                    ann_size=ann_size
                )
            else:
                # No annotations - empty features
                features = {
                    'pairs': [],
                    'scene': scene_classifier.classify(image),  # Still classify scene
                    'style': 'casual'  # Default
                }

            # Save to database
            cursor.execute('''
                INSERT INTO image_metadata (image_id, filename, pairs, scene, style)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(img_id),
                filename,
                json.dumps(features['pairs']),  # Store as JSON
                features['scene'],
                features['style']
            ))

            processed_count += 1

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped_count += 1
            continue

    db_conn.commit()
    db_conn.close()

    print(f"\nProcessed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    embeddings_matrix = np.vstack(all_embeddings).astype('float32')

    # Use HNSW index for fast approximate search
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 (connections per layer)
    index.hnsw.efConstruction = 200  # Quality of index construction
    index.hnsw.efSearch = 100  # Search quality

    # Add embeddings
    index.add(embeddings_matrix)

    # Save index
    print(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # Save image ID mapping (so we can map FAISS indices to image IDs)
    id_mapping_file = OUTPUT_DIR / "image_ids.json"
    with open(id_mapping_file, 'w') as f:
        json.dump(all_image_ids, f)

    print(f"\n" + "="*60)
    print("Indexing Complete!")
    print("="*60)
    print(f"FAISS index: {FAISS_INDEX_FILE}")
    print(f"Metadata DB: {METADATA_DB_FILE}")
    print(f"ID mapping: {id_mapping_file}")
    print(f"Total indexed: {len(all_image_ids)} images")


def main():
    print("="*60)
    print("Part A: Build Search Index")
    print("="*60)

    # Check dataset exists
    if not ANNOTATIONS_FILE.exists():
        print("\nError: Dataset not found!")
        print("Run: python scripts/download_fashionpedia.py")
        sys.exit(1)

    # Load dataset
    images_by_id, annotations_by_image = load_dataset()

    # Build index
    build_index(images_by_id, annotations_by_image)


if __name__ == "__main__":
    main()
