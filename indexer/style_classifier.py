import yaml
from pathlib import Path


class StyleClassifier:

    def __init__(self, config_path=None):
        # Load style rules from config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "extraction.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        style_config = config['style']
        self.formal_garments = set(style_config['formal_categories'])
        self.casual_garments = set(style_config['casual_categories'])
        self.formal_threshold = style_config['formal_threshold']

    def classify(self, garment_list):

        if not garment_list:
            return "casual"  # Default if no garments

        # Count formal garments
        formal_count = sum(1 for g in garment_list if self._is_formal_garment(g))
        total_count = len(garment_list)

        # Calculate ratio
        formal_ratio = formal_count / total_count

        # Classify based on threshold
        if formal_ratio >= self.formal_threshold:
            return "formal"
        else:
            return "casual"

    def _is_formal_garment(self, garment_name):
        # Normalize the garment name (handle different separators)
        garment_normalized = garment_name.lower().replace('-', '_').replace(' ', '_')

        # Check if it matches any formal category
        for formal_cat in self.formal_garments:
            formal_normalized = formal_cat.lower().replace('-', '_').replace(' ', '_')
            if formal_normalized in garment_normalized or garment_normalized in formal_normalized:
                return True

        return False


if __name__ == "__main__":
    print("Testing style classifier...")

    classifier = StyleClassifier()

    # Test formal outfit
    formal_outfit = ["blazer", "dress_shirt", "tie"]
    style = classifier.classify(formal_outfit)
    print(f"{formal_outfit} -> {style}")
    assert style == "formal"

    # Test casual outfit
    casual_outfit = ["t-shirt", "jeans", "sneaker"]
    style = classifier.classify(casual_outfit)
    print(f"{casual_outfit} -> {style}")
    assert style == "casual"

    # Test mixed outfit 
    mixed_outfit = ["blazer", "tie", "jeans", "t-shirt", "sneaker"]  # 2 formal out of 5
    style = classifier.classify(mixed_outfit)
    print(f"{mixed_outfit} -> {style} (2/5 = 40% formal)")
    assert style == "casual"

    print("Style classifier works!")
