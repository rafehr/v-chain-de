from scripts.filter_data_02 import get_ingredient_tags


def test_get_ingredient_tags_simple():
    """Tests a flat list of ingredients."""
    ingredients = [
        {"id": "en:sugar", "text": "Sucre"},
        {"id": "en:water", "text": "Wasser"},
    ]
    expected = ["en:sugar", "en:water"]
    assert get_ingredient_tags(ingredients) == expected


def test_get_ingredient_tags_nested():
    """Tests recursive extraction for nested ingredients (e.g., emulsifiers)."""
    ingredients = [
        {
            "id": "en:emulsifier",
            "text": "emulsifier",
            "ingredients": [
                {
                    "id": "en:e322",
                    "text": "lecithins",
                    "ingredients": [
                        {"id": "en:soya-lecithin", "text": "soya lecithin"}
                    ],
                }
            ],
        },
        {"id": "en:salt", "text": "sel"},
    ]
    expected = ["en:emulsifier", "en:e322", "en:soya-lecithin", "en:salt"]
    assert get_ingredient_tags(ingredients) == expected


def test_get_ingredient_tags_missing_id():
    """Tests that items without an 'id' are skipped, but sub-elements are still checked."""
    ingredients = [{"text": "Unknown mixture", "ingredients": [{"id": "en:herbs"}]}]
    assert get_ingredient_tags(ingredients) == ["en:herbs"]


def test_get_ingredient_tags_empty():
    """Tests behavior with empty lists."""
    assert get_ingredient_tags([]) == []
