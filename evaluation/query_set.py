# The 5 required assignment queries
ASSIGNMENT_QUERIES = [
    {
        'id': 'q1',
        'text': "A person in a bright yellow raincoat",
        'expected_attributes': {
            'colors': ['yellow'],
            'garments': ['raincoat', 'coat'],
            'scene': None,
            'style': None
        }
    },
    {
        'id': 'q2',
        'text': "Professional business attire inside a modern office",
        'expected_attributes': {
            'colors': [],
            'garments': ['suit', 'blazer', 'dress'],
            'scene': 'office',
            'style': 'formal'
        }
    },
    {
        'id': 'q3',
        'text': "Someone wearing a blue shirt sitting on a park bench",
        'expected_attributes': {
            'colors': ['blue'],
            'garments': ['shirt'],
            'scene': 'outdoor_nature',
            'style': None
        }
    },
    {
        'id': 'q4',
        'text': "Casual weekend outfit for a city walk",
        'expected_attributes': {
            'colors': [],
            'garments': ['jeans', 'tshirt', 't-shirt'],
            'scene': 'outdoor_urban',
            'style': 'casual'
        }
    },
    {
        'id': 'q5',
        'text': "A red tie and a white shirt in a formal setting",
        'expected_attributes': {
            'colors': ['red', 'white'],
            'garments': ['tie', 'shirt'],
            'scene': 'office',
            'style': 'formal'
        }
    }
]

# Additional test queries for completeness
ADDITIONAL_QUERIES = [
    {
        'id': 'q6',
        'text': "Blue jacket and red scarf",
        'expected_attributes': {
            'colors': ['blue', 'red'],
            'garments': ['jacket', 'scarf'],
            'scene': None,
            'style': None
        }
    },
    {
        'id': 'q7',
        'text': "Red jacket and blue scarf",  # Swap test
        'expected_attributes': {
            'colors': ['red', 'blue'],
            'garments': ['jacket', 'scarf'],
            'scene': None,
            'style': None
        }
    },
    {
        'id': 'q8',
        'text': "Cottagecore aesthetic",  # Pure vibe query
        'expected_attributes': {
            'colors': [],
            'garments': [],
            'scene': None,
            'style': 'casual'
        }
    },
    {
        'id': 'q9',
        'text': "Someone at a park",  # Scene-only query
        'expected_attributes': {
            'colors': [],
            'garments': [],
            'scene': 'outdoor_nature',
            'style': None
        }
    }
]

ALL_QUERIES = ASSIGNMENT_QUERIES + ADDITIONAL_QUERIES


def get_queries(include_additional=False):
    if include_additional:
        return ALL_QUERIES
    else:
        return ASSIGNMENT_QUERIES


def heuristic_relevance(image_metadata, query_attributes):
    score = 0
    max_score = 0

    # Check color-garment pairs
    if query_attributes['colors'] or query_attributes['garments']:
        max_score += 2
        # Check if any expected colors appear in image pairs
        image_pairs_str = ' '.join(image_metadata.get('pairs', []))

        for color in query_attributes['colors']:
            if color in image_pairs_str:
                score += 1

        for garment in query_attributes['garments']:
            if garment in image_pairs_str:
                score += 1

    # Check scene
    if query_attributes['scene'] is not None:
        max_score += 1
        if image_metadata.get('scene') == query_attributes['scene']:
            score += 1

    # Check style
    if query_attributes['style'] is not None:
        max_score += 1
        if image_metadata.get('style') == query_attributes['style']:
            score += 1

    # Consider relevant if at least 50% of expected attributes match
    if max_score == 0:
        return True  # No specific attributes, rely on embedding similarity

    return score >= max_score * 0.5


if __name__ == "__main__":
    print("Query Set Summary:")
    print("=" * 60)

    queries = get_queries(include_additional=True)
    print(f"Total queries: {len(queries)}")
    print(f"  - Assignment queries: {len(ASSIGNMENT_QUERIES)}")
    print(f"  - Additional queries: {len(ADDITIONAL_QUERIES)}")

    print("\nQuery Details:")
    for q in queries:
        print(f"\n[{q['id']}] {q['text']}")
        print(f"  Expected: {q['expected_attributes']}")
