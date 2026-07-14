import numpy as np
import yaml
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbone import BackboneModel


class SceneClassifier:
    def __init__(self, config_path=None, model=None):
        # Load scene configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "extraction.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.scene_config = config['scene']
        self.scene_classes = self.scene_config['classes']
        self.prompt_templates = self.scene_config['prompt_templates']

        # Load or reuse the backbone model
        self.model = model if model is not None else BackboneModel()

        # Pre-compute scene embeddings (so we don't recompute for every image)
        print("Pre-computing scene class embeddings...")
        self.scene_embeddings = self._compute_scene_embeddings()
        print(f"Ready to classify into {len(self.scene_classes)} scenes")

    def _compute_scene_embeddings(self):
        embeddings = {}
        for scene_class in self.scene_classes:
            # Get all prompt variations for this scene
            prompts = self.prompt_templates.get(scene_class, [f"a photo in {scene_class}"])

            # Embed each prompt
            prompt_embeddings = []
            for prompt in prompts:
                emb = self.model.embed_text(prompt)
                prompt_embeddings.append(emb)

            # Average them and normalize
            avg_embedding = np.mean(prompt_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            embeddings[scene_class] = avg_embedding

        return embeddings

    def classify(self, image):
        # Embed the image
        img_embedding = self.model.embed_image(image)

        # Compare to each scene class
        similarities = {}
        for scene_class, scene_embedding in self.scene_embeddings.items():
            similarity = np.dot(img_embedding, scene_embedding)
            similarities[scene_class] = similarity

        # Return the most similar scene
        best_scene = max(similarities, key=similarities.get)
        return best_scene


if __name__ == "__main__":
    print("Testing scene classifier...")

    classifier = SceneClassifier()

    # Test with a dummy image
    test_img = Image.new('RGB', (224, 224), color='gray')
    scene = classifier.classify(test_img)
    print(f"Test image classified as: {scene}")
    assert scene in classifier.scene_classes

    print("Scene classifier works!")
