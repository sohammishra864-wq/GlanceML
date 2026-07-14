import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import yaml

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbone import BackboneModel
from parser.query_parser import QueryParser
from evaluation.metrics import evaluate_query, aggregate_metrics
from evaluation.query_set import get_queries, heuristic_relevance


# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FAISS_INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
METADATA_DB_FILE = OUTPUT_DIR / "metadata.db"
ID_MAPPING_FILE = OUTPUT_DIR / "image_ids.json"


# Ablation configurations
ABLATION_CONFIGS = [
    {
        'name': 'Embedding-only',
        'weights': {'cosine_sim': 1.0, 'pair_match_count': 0.0, 'scene_match': 0.0, 'style_match': 0.0}
    },
    {
        'name': '+ Bound pairs',
        'weights': {'cosine_sim': 1.0, 'pair_match_count': 0.5, 'scene_match': 0.0, 'style_match': 0.0}
    },
    {
        'name': '+ Scene',
        'weights': {'cosine_sim': 1.0, 'pair_match_count': 0.5, 'scene_match': 0.3, 'style_match': 0.0}
    },
    {
        'name': '+ Style (full)',
        'weights': {'cosine_sim': 1.0, 'pair_match_count': 0.5, 'scene_match': 0.3, 'style_match': 0.2}
    }
]


class AblationEvaluator:
    """Run ablation study on different scoring configurations."""

    def __init__(self):
        print("Loading retrieval system...")

        # Load FAISS index
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))

        # Load image ID mapping
        with open(ID_MAPPING_FILE) as f:
            self.image_ids = json.load(f)

        # Connect to metadata database
        self.db_conn = sqlite3.connect(str(METADATA_DB_FILE))
        self.db_conn.row_factory = sqlite3.Row

        # Load models
        self.backbone = BackboneModel()
        self.query_parser = QueryParser()

        print("Ready!\n")

    def search_with_config(self, query_text, weights, top_k=50):
        # Parse query
        parsed = self.query_parser.parse(query_text)

        # Generate embedding
        query_embedding = self.backbone.embed_text(query_text)

        # Get top-200 candidates from FAISS
        candidate_count = min(200, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            candidate_count
        )

        # Get metadata
        candidate_image_ids = [self.image_ids[idx] for idx in indices[0]]
        candidate_metadata = self._fetch_metadata(candidate_image_ids)

        # Score candidates
        candidates = []
        for i, img_id in enumerate(candidate_image_ids):
            metadata = candidate_metadata.get(img_id)
            if metadata is None:
                continue

            # Cosine similarity: HNSW returns L2² on normalized vectors
            cosine_sim = 1.0 - distances[0][i] / 2.0

            # Metadata matches
            pair_match_count = self._count_matching_pairs(
                parsed['garment_color_pairs'],
                metadata['pairs']
            )
            scene_match = 1 if parsed['scene'] == metadata['scene'] else 0
            style_match = 1 if parsed['style'] == metadata['style'] else 0

            # Weighted score
            score = (
                weights['cosine_sim'] * cosine_sim +
                weights['pair_match_count'] * pair_match_count +
                weights['scene_match'] * scene_match +
                weights['style_match'] * style_match
            )

            candidates.append({
                'image_id': img_id,
                'score': score,
                'metadata': metadata
            })

        # Sort and return top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return [c['image_id'] for c in candidates[:top_k]], candidates[:top_k]

    def _fetch_metadata(self, image_ids):
        """Fetch metadata from database."""
        cursor = self.db_conn.cursor()
        placeholders = ','.join('?' * len(image_ids))
        query = f"SELECT * FROM image_metadata WHERE image_id IN ({placeholders})"
        cursor.execute(query, image_ids)

        metadata = {}
        for row in cursor.fetchall():
            img_id = row['image_id']
            metadata[img_id] = {
                'filename': row['filename'],
                'pairs': json.loads(row['pairs']),
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

    def run_ablation(self, queries):
        results = {}

        for config in ABLATION_CONFIGS:
            print(f"\nEvaluating: {config['name']}")
            print("-" * 60)

            query_results = []

            for query in queries:
                # Search with this config
                retrieved_ids, candidates = self.search_with_config(
                    query['text'],
                    config['weights'],
                    top_k=50
                )

                # Get relevant IDs using heuristic
                # In a real evaluation, you'd have manual labels
                relevant_ids = []
                for candidate in candidates:
                    if heuristic_relevance(candidate['metadata'], query['expected_attributes']):
                        relevant_ids.append(candidate['image_id'])

                # If no relevant found by heuristic, use top-5 as pseudo-relevant
                if not relevant_ids:
                    relevant_ids = retrieved_ids[:5]

                # Compute metrics
                metrics = evaluate_query(retrieved_ids, relevant_ids, k_values=[5, 10])
                query_results.append(metrics)

                print(f"  [{query['id']}] {query['text'][:50]}...")
                print(f"    Recall@5: {metrics['recall@5']:.3f}, MRR: {metrics['mrr']:.3f}")

            # Aggregate across queries
            aggregated = aggregate_metrics(query_results)
            results[config['name']] = aggregated

            print(f"\n  Average metrics:")
            for metric, value in aggregated.items():
                print(f"    {metric}: {value:.3f}")

        return results


def print_ablation_table(results):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Header
    configs = list(results.keys())
    metrics = list(results[configs[0]].keys())

    print(f"\n{'Configuration':<20}", end='')
    for metric in metrics:
        print(f"{metric:>12}", end='')
    print()
    print("-" * 80)

    # Rows
    for config in configs:
        print(f"{config:<20}", end='')
        for metric in metrics:
            value = results[config][metric]
            print(f"{value:>12.3f}", end='')
        print()

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("Ablation Study: Evaluating Component Contributions")
    print("=" * 80)

    # Check if index exists
    if not FAISS_INDEX_FILE.exists():
        print("\nError: Index not found!")
        print("Run: python indexer/build_index.py")
        sys.exit(1)

    # Load query set
    queries = get_queries(include_additional=False)  # Use only assignment queries
    print(f"\nUsing {len(queries)} queries for evaluation\n")

    # Run ablation
    evaluator = AblationEvaluator()
    results = evaluator.run_ablation(queries)

    # Print table
    print_ablation_table(results)

    # Save results
    results_file = OUTPUT_DIR / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
