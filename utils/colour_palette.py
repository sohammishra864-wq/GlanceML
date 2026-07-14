import numpy as np
from skimage import color as skcolor


# Extended fashion color palette (RGB values)
# Format: {name: (R, G, B)} in 0-255 range
FASHION_COLORS = {
    # Reds & Pinks
    "red": (255, 0, 0),
    "crimson": (220, 20, 60),
    "burgundy": (128, 0, 32),
    "maroon": (128, 0, 0),
    "coral": (255, 127, 80),
    "pink": (255, 192, 203),
    "rose": (255, 0, 127),
    "fuchsia": (255, 0, 255),
    "magenta": (255, 0, 255),
    "salmon": (250, 128, 114),

    # Oranges & Browns
    "orange": (255, 165, 0),
    "rust": (183, 65, 14),
    "terracotta": (204, 78, 92),
    "brown": (139, 69, 19),
    "tan": (210, 180, 140),
    "beige": (245, 245, 220),
    "khaki": (240, 230, 140),
    "camel": (193, 154, 107),
    "chocolate": (210, 105, 30),

    # Yellows & Golds
    "yellow": (255, 255, 0),
    "gold": (255, 215, 0),
    "mustard": (255, 219, 88),
    "cream": (255, 253, 208),
    "ivory": (255, 255, 240),
    "lemon": (255, 247, 0),

    # Greens
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "olive": (128, 128, 0),
    "forest": (34, 139, 34),
    "sage": (188, 184, 138),
    "mint": (189, 252, 201),
    "emerald": (80, 200, 120),
    "teal": (0, 128, 128),

    # Blues
    "blue": (0, 0, 255),
    "navy": (0, 0, 128),
    "royal": (65, 105, 225),
    "sky": (135, 206, 235),
    "turquoise": (64, 224, 208),
    "aqua": (0, 255, 255),
    "cyan": (0, 255, 255),
    "cobalt": (0, 71, 171),
    "indigo": (75, 0, 130),
    "denim": (21, 96, 189),

    # Purples
    "purple": (128, 0, 128),
    "violet": (238, 130, 238),
    "lavender": (230, 230, 250),
    "mauve": (224, 176, 255),
    "plum": (221, 160, 221),
    "lilac": (200, 162, 200),

    # Neutrals
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "charcoal": (54, 69, 79),
    "slate": (112, 128, 144),
    "ash": (178, 190, 181),

    # Metallics (approximate visual appearance)
    "bronze": (205, 127, 50),
    "copper": (184, 115, 51),
}


# Synonym normalization: maps variants to canonical names
COLOR_SYNONYMS = {
    # Red variants
    "crimson": "red",
    "burgundy": "red",
    "maroon": "red",
    "wine": "red",

    # Blue variants
    "navy": "blue",
    "royal": "blue",
    "denim": "blue",
    "cobalt": "blue",

    # Gray variants
    "grey": "gray",
    "charcoal": "gray",
    "silver": "gray",
    "slate": "gray",
    "ash": "gray",

    # Brown variants
    "tan": "brown",
    "khaki": "brown",
    "camel": "brown",
    "chocolate": "brown",

    # Yellow variants
    "mustard": "yellow",
    "gold": "yellow",
    "lemon": "yellow",

    # White variants
    "cream": "white",
    "ivory": "white",

    # Pink variants
    "rose": "pink",
    "salmon": "pink",

    # Cyan variants
    "aqua": "cyan",
    "turquoise": "cyan",

    # Magenta variants
    "fuchsia": "magenta",
}


def rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    """Convert RGB (0-255) to CIELAB."""
    rgb_normalized = np.array(rgb).reshape(1, 1, 3) / 255.0
    lab = skcolor.rgb2lab(rgb_normalized)
    return lab[0, 0]


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    CIEDE2000 perceptual color distance.

    Using skimage's deltaE_ciede2000 which implements the full formula.
    """
    return skcolor.deltaE_ciede2000(
        lab1.reshape(1, 1, 3),
        lab2.reshape(1, 1, 3)
    )[0, 0]


def get_nearest_color_name(rgb: tuple[int, int, int], palette: dict = None) -> str:
    if palette is None:
        palette = FASHION_COLORS

    target_lab = rgb_to_lab(rgb)

    min_distance = float('inf')
    nearest_name = None

    for name, color_rgb in palette.items():
        color_lab = rgb_to_lab(color_rgb)
        distance = ciede2000(target_lab, color_lab)

        if distance < min_distance:
            min_distance = distance
            nearest_name = name

    return nearest_name


def normalize_color_name(color_name: str) -> str:
    return COLOR_SYNONYMS.get(color_name.lower(), color_name.lower())


def get_palette_colors() -> list[str]:
    return sorted(FASHION_COLORS.keys())


if __name__ == "__main__":
    # Test
    print("Testing color matching...")

    # Bright red
    test_rgb = (255, 0, 0)
    name = get_nearest_color_name(test_rgb)
    print(f"RGB {test_rgb} → {name}")
    assert name == "red"

    # Dark blue
    test_rgb = (0, 0, 139)
    name = get_nearest_color_name(test_rgb)
    print(f"RGB {test_rgb} → {name}")
    assert name in ["navy", "blue"]

    # Normalization
    assert normalize_color_name("navy") == "blue"
    assert normalize_color_name("crimson") == "red"

    print(f"\n Color palette test passed!")
    print(f"  {len(FASHION_COLORS)} colors in palette")
    print(f"  {len(COLOR_SYNONYMS)} synonym mappings")
