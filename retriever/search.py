import argparse
import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbone import BackboneModel
from parser.query_parser import QueryParser


# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FAISS_INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
METADATA_DB_FILE = OUTPUT_DIR / "metadata.db"
ID_MAPPING_FILE = OUTPUT_DIR / "image_ids.json"
WEIGHTS_FILE = Path(__file__).parent.parent / "configs" / "weights.yaml"


class ImageRetriever:
    """Search engine for fashion images."""

    def __init__(self):
        """Load index, database, and models."""
        print("Loading retriever...")

        # Check files exist
        if not FAISS_INDEX_FILE.exists():
            print(f"Error: Index not found at {FAISS_INDEX_FILE}")
            print("Run: python indexer/build_index.py")
            sys.exit(1)

        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))

        # Load image ID mapping
        with open(ID_MAPPING_FILE) as f:
            self.image_ids = json.load(f)

        # Connect to metadata database
        self.db_conn = sqlite3.connect(str(METADATA_DB_FILE))
        self.db_conn.row_factory = sqlite3.Row  # Access columns by name

        # Load models
        print("Loading models...")
        self.backbone = BackboneModel()
        self.query_parser = QueryParser()

        # Load scoring weights
        with open(WEIGHTS_FILE) as f:
            weights_config = yaml.safe_load(f)
            self.weights = weights_config['weights']

        print("Retriever ready!\n")

    def search(self, query, top_k=10):
        print(f"Query: {query}")
        print("-" * 60)

        # Step 1: Parse query to structured constraints
        parsed = self.query_parser.parse(query)
        print(f"Parsed query:")
        print(f"  Garment-color pairs: {parsed['garment_color_pairs']}")
        print(f"  Scene: {parsed['scene']}")
        print(f"  Style: {parsed['style']}")

        # Step 2: Generate text embedding
        query_embedding = self.backbone.embed_text(query)

        # Step 3: Get top-200 candidates from FAISS (recall-first approach)
        # IMPORTANT: We don't filter by metadata here - that would hurt recall
        candidate_count = min(200, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            candidate_count
        )

        # Step 4: Get metadata for candidates
        candidate_image_ids = [self.image_ids[idx] for idx in indices[0]]
        candidate_metadata = self._fetch_metadata(candidate_image_ids)

        # Step 5: Compute reranking scores
        candidates = []
        for i, img_id in enumerate(candidate_image_ids):
            metadata = candidate_metadata.get(img_id)
            if metadata is None:
                continue

            # Cosine similarity: HNSW returns L2² on normalized vectors, L2² = 2 - 2*cos
            cosine_sim = 1.0 - distances[0][i] / 2.0

            # Compute metadata match scores
            pair_match_count = self._count_matching_pairs(
                parsed['garment_color_pairs'],
                metadata['pairs']
            )
            scene_match = 1 if parsed['scene'] == metadata['scene'] else 0
            style_match = 1 if parsed['style'] == metadata['style'] else 0

            # Weighted combination
            score = (
                self.weights['cosine_sim'] * cosine_sim +
                self.weights['pair_match_count'] * pair_match_count +
                self.weights['scene_match'] * scene_match +
                self.weights['style_match'] * style_match
            )

            candidates.append({
                'image_id': img_id,
                'filename': metadata['filename'],
                'score': score,
                'cosine_sim': cosine_sim,
                'matched_pairs': pair_match_count,
                'scene': metadata['scene'],
                'scene_match': scene_match,
                'style': metadata['style'],
                'style_match': style_match,
            })

        # Step 6: Sort by score and return top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        results = candidates[:top_k]

        return results

    def _fetch_metadata(self, image_ids):
        """Fetch metadata for a list of image IDs from database."""
        cursor = self.db_conn.cursor()

        # Query in batch
        placeholders = ','.join('?' * len(image_ids))
        query = f"SELECT * FROM image_metadata WHERE image_id IN ({placeholders})"
        cursor.execute(query, image_ids)

        # Build dict
        metadata = {}
        for row in cursor.fetchall():
            img_id = row['image_id']
            metadata[img_id] = {
                'filename': row['filename'],
                'pairs': json.loads(row['pairs']),  # Parse JSON
                'scene': row['scene'],
                'style': row['style']
            }

        return metadata

    def _count_matching_pairs(self, query_pairs, image_pairs):
        matches = 0
        for query_garment, query_color in query_pairs:
            for img_pair in image_pairs:
                if ':' not in img_pair:
                    continue
                img_garment, img_color = img_pair.rsplit(':', 1)
                if query_garment in img_garment and query_color == img_color:
                    matches += 1
                    break
        return matches


def print_results(results):
    """Pretty print search results."""
    print("\n" + "=" * 60)
    print(f"Top {len(results)} Results")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['filename']}")
        print(f"    Score: {result['score']:.3f}")
        print(f"    - Similarity: {result['cosine_sim']:.3f}")
        print(f"    - Matched pairs: {result['matched_pairs']}")
        print(f"    - Scene: {result['scene']} {'[MATCH]' if result['scene_match'] else ''}")
        print(f"    - Style: {result['style']} {'[MATCH]' if result['style_match'] else ''}")


def main():
    parser = argparse.ArgumentParser(description="Search fashion images")
    parser.add_argument('--query', type=str, required=True, help="Search query")
    parser.add_argument('--top_k', type=int, default=10, help="Number of results (default: 10)")

    args = parser.parse_args()

    # Create retriever
    retriever = ImageRetriever()

    # Search
    results = retriever.search(args.query, top_k=args.top_k)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
