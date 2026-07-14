#Prompt templates for scene classification and query parsing.

# Scene prompt templates (loaded from config at runtime, these are fallbacks)
DEFAULT_SCENE_PROMPTS = {
    "office": [
        "a photo in an office",
        "a photo in a modern office",
        "a photo in a corporate office",
        "a photo in a cubicle workspace",
    ],
    "home_interior": [
        "a photo at home",
        "a photo in a house interior",
        "a photo in a living room",
        "a photo in a bedroom",
    ],
    "outdoor_urban": [
        "a photo in a city",
        "a photo on a city street",
        "a photo in an urban area",
        "a photo in a downtown area",
    ],
    "outdoor_nature": [
        "a photo in nature",
        "a photo in a park",
        "a photo outdoors",
        "a photo in a natural setting",
    ],
}


# LLM query parser system prompt
QUERY_PARSER_SYSTEM = """You are a fashion search query parser. Extract structured attributes from natural language queries.

Extract:
- garment_color_pairs: List of {garment, color} objects (e.g., [{"garment": "shirt", "color": "white"}, {"garment": "tie", "color": "red"}])
- scene: One of [office, home_interior, outdoor_urban, outdoor_nature] or null
- style: One of [formal, casual] or null

Important:
- Preserve binding: "red tie and white shirt" → [{"garment":"tie","color":"red"}, {"garment":"shirt","color":"white"}]
- If no attributes mentioned, return empty lists/nulls (graceful degradation)
- Normalize colors to simple names (navy→blue, crimson→red)
"""


# LLM JSON schema for function calling
QUERY_PARSER_SCHEMA = {
    "name": "parse_fashion_query",
    "description": "Parse a fashion search query into structured attributes",
    "parameters": {
        "type": "object",
        "properties": {
            "garment_color_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "garment": {"type": "string"},
                        "color": {"type": "string"},
                    },
                    "required": ["garment", "color"],
                },
                "description": "List of garment-color pairs extracted from query",
            },
            "scene": {
                "type": ["string", "null"],
                "enum": ["office", "home_interior", "outdoor_urban", "outdoor_nature", None],
                "description": "Scene context if mentioned",
            },
            "style": {
                "type": ["string", "null"],
                "enum": ["formal", "casual", None],
                "description": "Style if mentioned",
            },
        },
        "required": ["garment_color_pairs", "scene", "style"],
    },
}
