import json
import random
import sqlite3
import sys
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import yaml

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbone import BackboneModel
from parser.query_parser import QueryParser
from evaluation.metrics import evaluate_query, aggregate_metrics
from evaluation.query_set import get_queries, heuristic_relevance


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FAISS_INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
METADATA_DB_FILE = OUTPUT_DIR / "metadata.db"
ID_MAPPING_FILE = OUTPUT_DIR / "image_ids.json"
WEIGHTS_FILE = Path(__file__).parent.parent / "configs" / "weights.yaml"
MODEL_FILE = OUTPUT_DIR / "reranker_model.txt"
RESULTS_FILE = OUTPUT_DIR / "reranker_comparison.json"


def generate_training_data(index, image_ids, db_conn, backbone, num_samples=100):
    print(f"Generating training data (up to {num_samples} images)...", flush=True)

    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM image_metadata WHERE pairs != '[]' LIMIT ?", (num_samples,))
    rows = cursor.fetchall()

    id_to_faiss_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    # Phase 1: Collect all template queries and metadata
    samples = []  # (img_embedding, pos_query, neg_query_or_None, pairs, scene, style)
    skipped = 0

    for row in rows:
        img_id = row['image_id']
        pairs = json.loads(row['pairs'])
        scene = row['scene']
        style = row['style']

        if not pairs:
            continue

        faiss_idx = id_to_faiss_idx.get(img_id)
        if faiss_idx is None:
            skipped += 1
            continue

        try:
            img_emb = index.reconstruct(faiss_idx).astype('float32')
        except RuntimeError:
            skipped += 1
            continue

        pos_query = _create_template_query(pairs, scene, style)
        neg_query = None
        if len(pairs) >= 2:
            swapped = _swap_colors(pairs)
            if swapped is not None:
                neg_query = _create_template_query(swapped, scene, style)

        samples.append((img_emb, pos_query, neg_query, pairs, scene, style))

    print(f"  Collected {len(samples)} samples, skipped {skipped}", flush=True)

    # Phase 2: Batch-embed all unique queries
    all_queries = []
    for img_emb, pos_q, neg_q, pairs, scene, style in samples:
        all_queries.append(pos_q)
        if neg_q:
            all_queries.append(neg_q)

    print(f"  Embedding {len(all_queries)} queries in batch...", flush=True)
    query_embeddings = _batch_embed_texts(backbone, all_queries)

    # Phase 3: Compute features using cached embeddings
    X = []
    y = []
    emb_idx = 0

    for img_emb, pos_q, neg_q, pairs, scene, style in samples:
        # Positive
        pos_emb = query_embeddings[emb_idx]
        emb_idx += 1
        cosine_sim = float(np.dot(pos_emb, img_emb))
        pair_match = _count_keyword_matches(pos_q, pairs)
        scene_match = 1 if (scene and scene.replace('_', ' ') in pos_q.lower()) else 0
        style_match = _check_style_match(style, pos_q)
        X.append([cosine_sim, pair_match, scene_match, style_match])
        y.append(1)

        # Negative
        if neg_q:
            neg_emb = query_embeddings[emb_idx]
            emb_idx += 1
            cosine_sim = float(np.dot(neg_emb, img_emb))
            pair_match = _count_keyword_matches(neg_q, pairs)
            scene_match = 1 if (scene and scene.replace('_', ' ') in neg_q.lower()) else 0
            style_match = _check_style_match(style, neg_q)
            X.append([cosine_sim, pair_match, scene_match, style_match])
            y.append(0)

    print(f"  Generated {len(X)} examples ({sum(y)} pos, {len(y)-sum(y)} neg)", flush=True)
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')


def _batch_embed_texts(backbone, texts, batch_size=32):
    import torch

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if i % (batch_size * 5) == 0:
            print(f"    Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...", flush=True)

        tokens = backbone.tokenizer(batch).to(backbone.device)
        with torch.no_grad():
            embs = backbone.model.encode_text(tokens)
        embs = embs.cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms
        all_embeddings.append(embs)

    return np.vstack(all_embeddings)


def _create_template_query(pairs, scene, style):
    parts = []
    for pair in pairs[:3]:
        if ':' in pair:
            garment, color = pair.rsplit(':', 1)
            parts.append(f"{color} {garment}")
    query = " and ".join(parts)
    if scene and scene != "unknown":
        query += f" in {scene.replace('_', ' ')}"
    if style == "formal":
        query += " professional"
    elif style == "casual":
        query += " casual"
    return query


def _swap_colors(pairs):
    parsed = [p.rsplit(':', 1) for p in pairs if ':' in p]
    if len(parsed) < 2:
        return pairs
    garments = [g for g, c in parsed]
    colors = [c for g, c in parsed]
    # Can't swap if all colors identical
    if len(set(colors)) < 2:
        return None
    shuffled = colors.copy()
    random.shuffle(shuffled)
    # Retry up to 10 times to get a different order
    for _ in range(10):
        if shuffled != colors:
            break
        random.shuffle(shuffled)
    else:
        return None
    return [f"{g}:{c}" for g, c in zip(garments, shuffled)]


def _count_keyword_matches(query_text, image_pairs):
    query_lower = query_text.lower()
    matches = 0
    for pair in image_pairs:
        if ':' not in pair:
            continue
        garment, color = pair.rsplit(':', 1)
        if garment.lower() in query_lower and color.lower() in query_lower:
            matches += 1
    return matches


def _check_style_match(image_style, query_text):
    q = query_text.lower()
    if image_style == "formal" and ("formal" in q or "professional" in q):
        return 1
    if image_style == "casual" and "casual" in q:
        return 1
    return 0


def train_model(X, y):
    print(f"\nTraining LightGBM on {len(X)} samples...", flush=True)

    train_data = lgb.Dataset(X, label=y, feature_name=[
        'cosine_sim', 'pair_match_count', 'scene_match', 'style_match'
    ])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
    }

    model = lgb.train(params, train_data, num_boost_round=100)
    print("Training complete.", flush=True)
    return model


def evaluate_with_model(model, index, image_ids, db_conn, backbone, queries, query_parser):
    print("\nEvaluating learned reranker...", flush=True)

    query_results = []

    for q in queries:
        query_text = q['text']
        parsed = query_parser.parse(query_text)

        query_embedding = backbone.embed_text(query_text)
        candidate_count = min(200, index.ntotal)
        distances, indices_arr = index.search(
            query_embedding.reshape(1, -1).astype('float32'), candidate_count
        )

        candidate_ids = [image_ids[idx] for idx in indices_arr[0]]

        cursor = db_conn.cursor()
        placeholders = ','.join('?' * len(candidate_ids))
        cursor.execute(
            f"SELECT * FROM image_metadata WHERE image_id IN ({placeholders})",
            candidate_ids
        )
        meta_map = {}
        for row in cursor.fetchall():
            meta_map[row['image_id']] = {
                'pairs': json.loads(row['pairs']),
                'scene': row['scene'],
                'style': row['style'],
            }

        scored = []
        for i, img_id in enumerate(candidate_ids):
            meta = meta_map.get(img_id)
            if meta is None:
                continue

            cosine_sim = 1.0 - distances[0][i] / 2.0
            pair_match = _count_query_pair_matches(parsed['garment_color_pairs'], meta['pairs'])
            scene_match = 1 if parsed['scene'] == meta['scene'] else 0
            style_match = 1 if parsed['style'] == meta['style'] else 0

            features = np.array([[cosine_sim, pair_match, scene_match, style_match]], dtype='float32')
            lgb_score = model.predict(features)[0]
            scored.append((img_id, lgb_score, meta))

        scored.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [s[0] for s in scored]

        relevant_ids = [s[0] for s in scored if heuristic_relevance(s[2], q['expected_attributes'])]
        if not relevant_ids:
            relevant_ids = retrieved_ids[:5]

        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values=[5, 10])
        query_results.append(metrics)
        print(f"  [{q['id']}] Recall@5={metrics['recall@5']:.2f} MRR={metrics['mrr']:.3f}", flush=True)

    return aggregate_metrics(query_results)


def evaluate_weighted_baseline(index, image_ids, db_conn, backbone, queries, query_parser, weights):
    print("\nEvaluating weighted baseline...", flush=True)

    query_results = []

    for q in queries:
        query_text = q['text']
        parsed = query_parser.parse(query_text)

        query_embedding = backbone.embed_text(query_text)
        candidate_count = min(200, index.ntotal)
        distances, indices_arr = index.search(
            query_embedding.reshape(1, -1).astype('float32'), candidate_count
        )

        candidate_ids = [image_ids[idx] for idx in indices_arr[0]]

        cursor = db_conn.cursor()
        placeholders = ','.join('?' * len(candidate_ids))
        cursor.execute(
            f"SELECT * FROM image_metadata WHERE image_id IN ({placeholders})",
            candidate_ids
        )
        meta_map = {}
        for row in cursor.fetchall():
            meta_map[row['image_id']] = {
                'pairs': json.loads(row['pairs']),
                'scene': row['scene'],
                'style': row['style'],
            }

        scored = []
        for i, img_id in enumerate(candidate_ids):
            meta = meta_map.get(img_id)
            if meta is None:
                continue

            cosine_sim = 1.0 - distances[0][i] / 2.0
            pair_match = _count_query_pair_matches(parsed['garment_color_pairs'], meta['pairs'])
            scene_match = 1 if parsed['scene'] == meta['scene'] else 0
            style_match = 1 if parsed['style'] == meta['style'] else 0

            score = (
                weights['cosine_sim'] * cosine_sim +
                weights['pair_match_count'] * pair_match +
                weights['scene_match'] * scene_match +
                weights['style_match'] * style_match
            )
            scored.append((img_id, score, meta))

        scored.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [s[0] for s in scored]

        relevant_ids = [s[0] for s in scored if heuristic_relevance(s[2], q['expected_attributes'])]
        if not relevant_ids:
            relevant_ids = retrieved_ids[:5]

        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values=[5, 10])
        query_results.append(metrics)
        print(f"  [{q['id']}] Recall@5={metrics['recall@5']:.2f} MRR={metrics['mrr']:.3f}", flush=True)

    return aggregate_metrics(query_results)


def _count_query_pair_matches(query_pairs, image_pairs):
    matches = 0
    for qg, qc in query_pairs:
        for ip in image_pairs:
            if ':' not in ip:
                continue
            ig, ic = ip.rsplit(':', 1)
            if qg in ig and qc == ic:
                matches += 1
                break
    return matches


def main():
    print("=" * 60, flush=True)
    print("Step 7: Learned Reranker Experiment", flush=True)
    print("=" * 60, flush=True)

    if not FAISS_INDEX_FILE.exists():
        print("\nError: Index not found. Run: python indexer/build_index.py")
        sys.exit(1)

    # Load everything
    print("\nLoading index and models...", flush=True)
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(ID_MAPPING_FILE) as f:
        image_ids = json.load(f)

    db_conn = sqlite3.connect(str(METADATA_DB_FILE))
    db_conn.row_factory = sqlite3.Row

    backbone = BackboneModel()
    query_parser = QueryParser()

    with open(WEIGHTS_FILE) as f:
        weights = yaml.safe_load(f)['weights']

    # Generate training data (batched for speed)
    random.seed(42)
    X_train, y_train = generate_training_data(index, image_ids, db_conn, backbone, num_samples=100)

    if len(X_train) < 10:
        print("Error: Not enough training data generated.")
        sys.exit(1)

    # Train
    model = train_model(X_train, y_train)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    print(f"Model saved to {MODEL_FILE}", flush=True)

    # Evaluate both on the same query set
    queries = get_queries(include_additional=False)

    learned_metrics = evaluate_with_model(
        model, index, image_ids, db_conn, backbone, queries, query_parser
    )
    baseline_metrics = evaluate_weighted_baseline(
        index, image_ids, db_conn, backbone, queries, query_parser, weights
    )

    # Print comparison
    print("\n" + "=" * 70, flush=True)
    print("RESULTS: Weighted Baseline vs Learned Reranker", flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'Metric':<15} {'Baseline':>12} {'Learned':>12} {'Delta':>12}", flush=True)
    print("-" * 55, flush=True)

    deltas = []
    for metric in sorted(baseline_metrics.keys()):
        b = baseline_metrics[metric]
        l = learned_metrics[metric]
        d = l - b
        deltas.append(d)
        sign = "+" if d >= 0 else ""
        print(f"{metric:<15} {b:>12.3f} {l:>12.3f} {sign}{d:>11.3f}", flush=True)

    avg_delta = np.mean(deltas)

    # Decision
    print("\n" + "=" * 70, flush=True)
    THRESHOLD = 0.02
    if avg_delta > THRESHOLD:
        decision = "PROMOTE learned reranker to default"
        reason = f"Average improvement {avg_delta:.3f} exceeds threshold {THRESHOLD}"
    else:
        decision = "KEEP weighted baseline as default"
        reason = (
            f"Average delta {avg_delta:.3f} does not exceed threshold {THRESHOLD}. "
            "Learned model adds complexity without clear gain — valid negative finding."
        )

    print(f"DECISION: {decision}", flush=True)
    print(f"Reason: {reason}", flush=True)
    print("=" * 70, flush=True)

    # Save results
    results = {
        'baseline_metrics': {k: float(v) for k, v in baseline_metrics.items()},
        'learned_metrics': {k: float(v) for k, v in learned_metrics.items()},
        'decision': decision,
        'reason': reason,
        'avg_delta': float(avg_delta),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    db_conn.close()


if __name__ == "__main__":
    main()
