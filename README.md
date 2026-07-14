# GlanceML

Fashion-Aware Compositional Image Retrieval

Retrieve fashion images using natural language queries while preserving attribute-object binding (e.g., "red tie + white shirt" ≠ "white tie + red shirt").

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Automatic (recommended):**
```bash
python scripts/download_fashionpedia.py
```

**Manual (if automatic fails):**
1. Download annotations:
   - [Validation annotations](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json)
   - Save to: `data/raw/annotations/val_annotations.json`

2. Download images:
   - [COCO val2017 images](http://images.cocodataset.org/zips/val2017.zip) (~1GB)
   - Extract to: `data/raw/images/val2017/`

3. Validate:
```bash
python scripts/validate_dataset.py
```

### 3. Test Backbone Model

```bash
python models/backbone.py
```

This confirms `Marqo/marqo-fashionSigLIP` loads correctly and produces embeddings.

## Architecture

See [docs/architecture.md](docs/architecture.md) for full system design.

**Key Innovation**: Bound `garment:color` tokens + reranking (not hard filters) preserve compositional correctness while maintaining high recall.

**Pipeline**:
- **Offline**: Garment detection → color/scene/style extraction → embedding + FAISS index → SQLite metadata
- **Online**: Query parsing → FAISS top-200 → reranking on bound attributes → explainable results

## Project Structure

```
GlanceML/
├── indexer/       # Extraction & index building
├── retriever/     # Query-time search
├── parser/        # Query → structured constraints
├── models/        # Backbone wrapper (swappable)
├── utils/         # Color palette, synonyms, prompts
├── evaluation/    # Metrics & ablation harness
├── configs/       # Weights, model paths, extraction params
├── scripts/       # Dataset download & validation
└── docs/          # Architecture & implementation plan
```


## Usage

### Part A: Build Index

```bash
# After dataset download completes
python indexer/build_index.py
```

Outputs:
- `outputs/faiss_index.bin` - FAISS similarity search index
- `outputs/metadata.db` - SQLite database with garment colors, scene, style
- `outputs/image_ids.json` - Mapping from FAISS indices to image IDs

### Part B: Search

```bash
# Example queries
python retriever/search.py --query "red tie and white shirt in an office"
python retriever/search.py --query "casual weekend outfit" --top_k 10
python retriever/search.py --query "bright yellow raincoat"
```

Returns ranked list of matching images with scores and matched attributes.

## GPU Support

Default: `faiss-cpu`. For GPU, edit `requirements.txt`:
```bash
# Comment out faiss-cpu, uncomment:
# faiss-gpu>=1.7.4
```

## License & Citation

Fashionpedia dataset: See [official page](https://fashionpedia.github.io/home/index.html) for terms.