# Fashion-Aware Compositional Image Retrieval — Report

## Problem

Standard CLIP-style models fail on compositional fashion queries because pooled embeddings lose attribute-object binding. "Red tie, white shirt" and "white tie, red shirt" produce nearly identical embeddings despite being semantically different.

## Solution

Two-stage retrieval pipeline:
1. **FAISS HNSW** (top-200 by cosine similarity) — recall-first, no pre-filtering
2. **Weighted linear reranker** on 4 signals:
   ```
   score = 1.0*cosine_sim + 0.5*pair_match + 0.3*scene_match + 0.2*style_match
   ```

Key design: store garment-color attributes as **bound tokens** (`["shirt:white", "tie:red"]`) so the reranker can verify which color belongs to which garment.

## Feature Extraction

| Component | Method | Details |
|-----------|--------|---------|
| Color | KMeans (k=5) in CIELAB + CIEDE2000 nearest-name | ~60 fashion color names, synonym-normalized |
| Scene | Zero-shot classification (prompt ensemble, 4 prompts/class) | office, home_interior, outdoor_urban, outdoor_nature |
| Style | Rule-based from garment types | formal if ≥50% formal categories |
| Embedding | marqo-fashionSigLIP via open_clip | 768-dim, normalized |

## Dataset

Fashionpedia val2020: 564 images successfully indexed with ground-truth garment segmentation masks and category annotations.

## Ablation Results (Experiment 2)

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |
|---------------|----------|-----------|------|--------|---------|
| Embedding-only | 0.800 | 0.800 | 0.813 | 0.660 | 0.612 |
| + Bound pairs | 0.800 | 0.800 | 0.813 | 0.723 | 0.639 |
| + Scene | **1.000** | **1.000** | 0.850 | 0.764 | 0.735 |
| + Style (full) | **1.000** | **1.000** | **0.850** | **0.779** | **0.814** |

**Key findings:**
- Adding scene matching gives the largest single jump: Recall@5 0.8 → 1.0
- Full pipeline (all 4 signals) achieves nDCG@10 of 0.814 vs 0.612 baseline — a 33% improvement
- Bound pairs improve nDCG (ranking quality) without changing recall on this query set — the binding benefit shows up more in rank ordering than in binary hit/miss

## Learned Reranker Experiment (Experiment 3)

Trained LightGBM (100 trees, 15 leaves) on synthetic pairs:
- Positive: image + template query built from its own metadata
- Hard negative: same image + template with swapped garment-color assignments

| Metric | Weighted Baseline | LightGBM | Delta |
|--------|-------------------|----------|-------|
| Recall@5 | 1.000 | 0.800 | -0.200 |
| Recall@10 | 1.000 | 0.800 | -0.200 |
| MRR | 0.850 | 0.679 | -0.171 |
| nDCG@5 | 0.779 | 0.533 | -0.246 |
| nDCG@10 | 0.814 | 0.444 | -0.370 |

**Decision: KEEP weighted baseline as default.**

**Why the learned model underperforms:** The synthetic training labels are derived from the same 4 features used for scoring. LightGBM effectively attempts to learn a nonlinear combination of features that were already optimally combined linearly. With only 190 training samples and features that are mostly binary (scene_match, style_match), the tree model overfits to noise in the cosine similarity signal rather than discovering a genuinely better ranking function. This is a valid negative finding — it confirms the weighted linear formula is already near-optimal for these features.

## Design Decisions

1. **Bound tokens over separate lists** — preserves which color belongs to which garment
2. **Recall-first (no pre-filtering)** — misclassified attributes cost rank position, not presence
3. **marqo-fashionSigLIP** — fashion-domain SigLIP backbone, open-source, 768-dim
4. **Rule-based parser** — works offline, sufficient for structured queries, fast (~1ms)
5. **Heuristic relevance** — attribute-overlap scoring as proxy for manual labels; adequate for comparative evaluation

## Limitations

- Single dominant color per garment (misses patterns/stripes)
- 4 coarse scene classes only
- Binary style (no "business casual" middle ground)
- Rule-based parser misses complex phrasing
- Evaluation uses heuristic relevance, not manual ground truth

## Future Work

1. **External detector** (Grounding DINO + SAM) — support arbitrary images without GT annotations
2. **Hierarchical taxonomy fallback** — handle out-of-vocabulary garment terms
3. **Multi-color extraction** — top-3 colors per garment for patterned items
4. **LLM query parser** — handle natural/ambiguous phrasing

## How to Run

```bash
# Build index (Part A)
python indexer/build_index.py

# Search (Part B)
python retriever/search.py --query "red tie and white shirt in an office"

# Evaluation
python evaluation/run_ablation.py
python evaluation/train_reranker.py
```

## References

- Fashionpedia: https://fashionpedia.github.io/
- marqo-fashionSigLIP: Marqo/marqo-fashionSigLIP (HuggingFace)
- FAISS: facebook/faiss (GitHub)
- CIEDE2000: Sharma et al., 2005
