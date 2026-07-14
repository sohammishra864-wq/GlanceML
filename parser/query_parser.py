import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.color_palette import get_palette_colors, normalize_color_name

GARMENT_KEYWORDS = [
    # Tops
    'shirt', 't-shirt', 'tshirt', 'blouse', 'top', 'sweater', 'cardigan',
    'hoodie', 'sweatshirt', 'tank', 'vest',
    # Bottoms
    'pants', 'jeans', 'trousers', 'skirt', 'shorts', 'leggings',
    # Outerwear
    'jacket', 'blazer', 'coat', 'raincoat', 'windbreaker', 'parka',
    'trench', 'overcoat',
    # Dresses/Suits
    'dress', 'suit', 'gown', 'jumpsuit', 'romper',
    # Accessories
    'tie', 'scarf', 'hat', 'cap', 'beanie', 'gloves', 'belt',
    # Footwear
    'shoes', 'sneakers', 'boots', 'sandals', 'heels', 'flats', 'loafers'
]

# Scene keywords (expanded)
SCENE_KEYWORDS = {
    'office': [
        'office', 'corporate', 'workplace', 'desk', 'cubicle',
        'work', 'meeting', 'boardroom', 'conference'
    ],
    'home_interior': [
        'home', 'house', 'indoors', 'interior', 'living room', 'bedroom',
        'cozy', 'lounging', 'relaxing at home', 'staying in'
    ],
    'outdoor_urban': [
        'city', 'street', 'urban', 'downtown', 'sidewalk',
        'shopping', 'cafe', 'restaurant', 'walking around town',
        'city walk', 'errands'
    ],
    'outdoor_nature': [
        'park', 'nature', 'outdoors', 'forest', 'garden', 'bench',
        'hiking', 'trail', 'beach', 'lake', 'picnic', 'camping'
    ]
}

# Style keywords (expanded with vibes/aesthetics)
STYLE_KEYWORDS = {
    'formal': [
        'formal', 'professional', 'business', 'suit', 'dressed up',
        'elegant', 'sophisticated', 'polished', 'smart', 'dressy',
        'office wear', 'work attire', 'business casual'
    ],
    'casual': [
        'casual', 'weekend', 'relaxed', 'everyday', 'laid back',
        'comfortable', 'easy', 'chill', 'informal', 'day off',
        # Aesthetics that map to casual
        'cottagecore', 'cozy', 'comfy', 'athleisure', 'streetwear',
        'bohemian', 'boho', 'minimalist', 'edgy', 'grunge',
        'vintage', 'retro', 'indie', 'artsy'
    ]
}

# Weather/occasion mappings (help infer style/scene/colors)
WEATHER_OCCASION_HINTS = {
    # Weather
    'rainy': {'garments': ['raincoat', 'coat', 'jacket'], 'colors': ['gray', 'black', 'navy']},
    'rain': {'garments': ['raincoat', 'coat', 'jacket'], 'colors': ['gray', 'black', 'navy']},
    'sunny': {'garments': ['dress', 'shorts', 't-shirt'], 'colors': ['yellow', 'white', 'pink']},
    'summer': {'garments': ['shorts', 'dress', 't-shirt', 'sandals'], 'colors': ['white', 'blue', 'yellow']},
    'winter': {'garments': ['coat', 'sweater', 'boots'], 'colors': ['black', 'gray', 'brown']},
    'cold': {'garments': ['sweater', 'coat', 'jacket'], 'colors': ['gray', 'black', 'brown']},

    # Occasions
    'date': {'style': 'formal', 'garments': ['dress', 'blazer']},
    'date night': {'style': 'formal', 'garments': ['dress', 'blazer']},
    'party': {'style': 'formal', 'garments': ['dress', 'suit']},
    'gym': {'style': 'casual', 'garments': ['shorts', 'hoodie', 'sneakers']},
    'workout': {'style': 'casual', 'garments': ['shorts', 'hoodie', 'sneakers']},
    'beach': {'style': 'casual', 'garments': ['shorts', 'dress', 'sandals'], 'scene': 'outdoor_nature'},
    'brunch': {'style': 'casual', 'garments': ['dress', 'jeans'], 'scene': 'outdoor_urban'},
    'coffee': {'style': 'casual', 'scene': 'outdoor_urban'},
}


class QueryParser:
    """Parse natural language queries into structured constraints."""

    def __init__(self):
        self.color_names = get_palette_colors()

    def parse(self, query):
        """
        Parse a natural language query.

        Returns dict with:
        - garment_color_pairs: list of (garment, color) tuples
        - scene: string or None
        - style: string or None
        """
        query_lower = query.lower()

        # Check for weather/occasion hints first
        hints = self._extract_hints(query_lower)

        # Extract explicit attributes
        pairs = self._extract_garment_color_pairs(query_lower)
        scene = self._extract_scene(query_lower)
        style = self._extract_style(query_lower)

        # Apply hints if no explicit attributes found
        if not pairs and 'garments' in hints:
            # Add suggested garments (without colors, let embedding handle it)
            pairs = [(g, None) for g in hints['garments'][:2]]  # Top 2 suggestions
            # Filter out None colors for final output
            pairs = [(g, c) for g, c in pairs if c is not None]

        if not scene and 'scene' in hints:
            scene = hints['scene']

        if not style and 'style' in hints:
            style = hints['style']

        return {
            'garment_color_pairs': pairs,
            'scene': scene,
            'style': style
        }

    def _extract_hints(self, query):
        """
        Extract hints from weather/occasion keywords.

        Example: "rainy day outfit" -> suggests raincoat, coat colors
        """
        hints = {}

        for keyword, suggestion in WEATHER_OCCASION_HINTS.items():
            if keyword in query:
                # Merge suggestions
                if 'garments' in suggestion:
                    hints.setdefault('garments', []).extend(suggestion['garments'])
                if 'colors' in suggestion:
                    hints.setdefault('colors', []).extend(suggestion['colors'])
                if 'style' in suggestion:
                    hints['style'] = suggestion['style']
                if 'scene' in suggestion:
                    hints['scene'] = suggestion['scene']

        # Remove duplicates
        if 'garments' in hints:
            hints['garments'] = list(dict.fromkeys(hints['garments']))
        if 'colors' in hints:
            hints['colors'] = list(dict.fromkeys(hints['colors']))

        return hints

    def _extract_garment_color_pairs(self, query):
      
        pairs = []

        # Look for "color garment" patterns
        for color in self.color_names:
            for garment in GARMENT_KEYWORDS:
                # Pattern: "color garment" (e.g., "red tie")
                if f"{color} {garment}" in query:
                    normalized_color = normalize_color_name(color)
                    pairs.append((garment, normalized_color))

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for g, c in pairs:
            key = (g, c)
            if key not in seen:
                seen.add(key)
                unique_pairs.append((g, c))

        return unique_pairs

    def _extract_scene(self, query):
        """Extract scene class from query."""
        for scene_class, keywords in SCENE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    return scene_class
        return None

    def _extract_style(self, query):
        """Extract style from query."""
        for style_name, keywords in STYLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    return style_name
        return None



#def parse_with_llm(query, api_key=None, provider='openai'): 
 #   parser = QueryParser()
  #  return parser.parse(query) optional right now only rule based parsing


if __name__ == "__main__":  # Test the parser
    parser = QueryParser()

    test_queries = [
        # Explicit attribute queries
        "red tie and white shirt",
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",

        # Vibe/aesthetic queries
        "cozy rainy day outfit",
        "cottagecore aesthetic",
        "edgy streetwear look",

        # Occasion queries 
        "date night outfit",
        "gym workout clothes",
        "beach day look",
        "coffee shop casual",

        # Weather queries 
        "rainy day aesthetic",
        "summer vibes",
        "winter layers"
    ]

    print("Testing Enhanced Query Parser:")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = parser.parse(query)
        print(f"  Pairs: {result['garment_color_pairs']}")
        print(f"  Scene: {result['scene']}")
        print(f"  Style: {result['style']}")

    print("\n" + "=" * 60)
    print("Enhanced parser works!")
    print("\nNow handles:")
    print("  - Explicit attributes (red tie, white shirt)")
    print("  - Vibe/aesthetic queries (cozy, cottagecore)")
    print("  - Occasion queries (date night, gym)")
    print("  - Weather queries (rainy day, summer)")
