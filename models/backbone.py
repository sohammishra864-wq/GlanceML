"""
Backbone model for embedding images and text.

Uses marqo-fashionSigLIP via open_clip - simpler and more reliable.
"""
import numpy as np
import torch
from PIL import Image
import open_clip


class BackboneModel:
    """Handles embedding images and text queries."""

    def __init__(self, model_name="hf-hub:Marqo/marqo-fashionSigLIP", device=None):
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model on {self.device}...")

        # Load model using open_clip (simpler than transformers for this model)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        print("Model loaded!")

    @torch.no_grad()
    def embed_image(self, image): # image to vec converter
        # Preprocess and move to device
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get embedding
        embedding = self.model.encode_image(image_tensor)
        embedding = embedding.cpu().numpy()[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @torch.no_grad()
    def embed_text(self, text):
        # Tokenize
        text_tokens = self.tokenizer([text]).to(self.device)

        # Get embedding
        embedding = self.model.encode_text(text_tokens)
        embedding = embedding.cpu().numpy()[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @torch.no_grad()
    def embed_images_batch(self, images, batch_size=32):
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Preprocess batch
            image_tensors = torch.stack([self.preprocess(img) for img in batch])
            image_tensors = image_tensors.to(self.device)

            # Get embeddings
            embeddings = self.model.encode_image(image_tensors)
            embeddings = embeddings.cpu().numpy()

            # Normalize each
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


def test_model():
    print("Testing backbone model...")

    model = BackboneModel()

    # Test image embedding
    test_img = Image.new('RGB', (224, 224), color='red')
    img_emb = model.embed_image(test_img)
    print(f"Image embedding shape: {img_emb.shape}")
    assert len(img_emb.shape) == 1 and img_emb.shape[0] > 0
    assert abs(np.linalg.norm(img_emb) - 1.0) < 0.01

    # Test text embedding
    text_emb = model.embed_text("red dress")
    print(f"Text embedding shape: {text_emb.shape}")
    assert text_emb.shape == img_emb.shape
    assert abs(np.linalg.norm(text_emb) - 1.0) < 0.01

    # Test similarity
    similarity = np.dot(img_emb, text_emb)
    print(f"Similarity: {similarity:.3f}")

    print("Backbone model works!")


if __name__ == "__main__":
    test_model()
