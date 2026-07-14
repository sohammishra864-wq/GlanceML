import json
import sqlite3
import sys
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backbone import BackboneModel
from parser.query_parser import QueryParser

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FAISS_INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
METADATA_DB_FILE = OUTPUT_DIR / "metadata.db"
ID_MAPPING_FILE = OUTPUT_DIR / "image_ids.json"
WEIGHTS_FILE = PROJECT_ROOT / "configs" / "weights.yaml"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "fashionpedia"


def load_system():
    """Load all components once at startup."""
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(ID_MAPPING_FILE) as f:
        image_ids = json.load(f)
    backbone = BackboneModel()
    query_parser = QueryParser()
    with open(WEIGHTS_FILE) as f:
        weights = yaml.safe_load(f)['weights']
    return index, image_ids, backbone, query_parser, weights


def get_db():
    """Get a thread-local SQLite connection."""
    conn = sqlite3.connect(str(METADATA_DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


print("Loading GlanceML system...", flush=True)
INDEX, IMAGE_IDS, BACKBONE, QUERY_PARSER, WEIGHTS = load_system()
DB_CONN = get_db()
print("Ready!", flush=True)


def search(query, top_k=8): # run  full retrieval pipeline and return images & info
    if not query.strip():
        return [], ""

    parsed = QUERY_PARSER.parse(query)

    query_embedding = BACKBONE.embed_text(query)
    candidate_count = min(200, INDEX.ntotal)
    distances, indices = INDEX.search(
        query_embedding.reshape(1, -1).astype('float32'), candidate_count
    )

    candidate_ids = [IMAGE_IDS[idx] for idx in indices[0]]

    cursor = DB_CONN.cursor()
    placeholders = ','.join('?' * len(candidate_ids))
    cursor.execute(
        f"SELECT * FROM image_metadata WHERE image_id IN ({placeholders})",
        candidate_ids
    )
    meta_map = {}
    for row in cursor.fetchall():
        meta_map[row['image_id']] = {
            'filename': row['filename'],
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

        pair_match = 0
        for qg, qc in parsed['garment_color_pairs']:
            for ip in meta['pairs']:
                if ':' not in ip:
                    continue
                ig, ic = ip.rsplit(':', 1)
                if qg in ig and qc == ic:
                    pair_match += 1
                    break

        scene_match = 1 if parsed['scene'] == meta['scene'] else 0
        style_match = 1 if parsed['style'] == meta['style'] else 0

        score = (
            WEIGHTS['cosine_sim'] * cosine_sim +
            WEIGHTS['pair_match_count'] * pair_match +
            WEIGHTS['scene_match'] * scene_match +
            WEIGHTS['style_match'] * style_match
        )

        scored.append({
            'image_id': img_id,
            'filename': meta['filename'],
            'score': score,
            'cosine_sim': cosine_sim,
            'pair_match': pair_match,
            'scene': meta['scene'],
            'scene_match': scene_match,
            'style': meta['style'],
            'style_match': style_match,
            'pairs': meta['pairs'],
        })

    scored.sort(key=lambda x: x['score'], reverse=True)
    results = scored[:top_k]

    # Build gallery
    gallery_items = []
    for r in results:
        img_path = IMAGES_DIR / r['filename']
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            caption = f"Score: {r['score']:.2f} | {r['scene']} | {r['style']}"
            if r['pairs']:
                caption += f"\n{', '.join(r['pairs'][:4])}"
            gallery_items.append((img, caption))

    # Build details text
    details = f"**Parsed query:**\n"
    details += f"- Pairs: {parsed['garment_color_pairs']}\n"
    details += f"- Scene: {parsed['scene'] or '—'}\n"
    details += f"- Style: {parsed['style'] or '—'}\n\n"
    details += f"**Top {len(results)} results** (from {INDEX.ntotal} indexed images)\n\n"

    for i, r in enumerate(results, 1):
        match_tags = []
        if r['pair_match'] > 0:
            match_tags.append(f"{r['pair_match']} pair(s) matched")
        if r['scene_match']:
            match_tags.append("scene matched")
        if r['style_match']:
            match_tags.append("style matched")
        match_str = " | ".join(match_tags) if match_tags else "embedding only"
        details += f"{i}. **{r['filename']}** — score {r['score']:.3f} ({match_str})\n"

    return gallery_items, details


EXAMPLE_QUERIES = [
    "red tie and white shirt in a formal setting",
    "bright yellow raincoat",
    "professional business attire in an office",
    "blue shirt sitting on a park bench",
    "casual weekend outfit for a city walk",
    "cozy rainy day outfit",
    "black dress formal",
]

CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 0.95em;
    margin-bottom: 1.5em;
}
"""

with gr.Blocks(title="GlanceML") as demo:

    gr.HTML("<h1 class='main-title'>GlanceML</h1>")
    gr.HTML("<p class='subtitle'>Fashion-Aware Compositional Image Retrieval</p>")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                placeholder="Describe what you're looking for...",
                label="Search Query",
                lines=1,
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                minimum=4, maximum=20, value=8, step=1, label="Results"
            )

    search_btn = gr.Button("Search", variant="primary", size="lg")

    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUERIES],
        inputs=[query_input],
        label="Try these queries",
    )

    gallery = gr.Gallery(
        label="Results",
        columns=4,
        height=480,
        object_fit="cover",
    )

    details = gr.Markdown(label="Details")

    search_btn.click(fn=search, inputs=[query_input, top_k_slider], outputs=[gallery, details])
    query_input.submit(fn=search, inputs=[query_input, top_k_slider], outputs=[gallery, details])

if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CSS,
    )
