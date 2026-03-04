import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import orjson
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from tqdm import tqdm

load_dotenv()

DATA_PATH = os.getenv("OFF_DATA_PATH")
if not DATA_PATH:
    raise ValueError("Data path not set in .env.")

FILTERED_DATA_PATH = os.getenv("FILTERED_OFF_DATA_PATH")
if not FILTERED_DATA_PATH:
    raise ValueError("Filtered data path not set in .env.")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / FILTERED_DATA_PATH
DATA_FILE = BASE_DIR / DATA_PATH


class TaxonomyTag(BaseModel):
    """
    Represents a taxonomy tag (could be for an ingredient, a category, etc.)
    from the OpenFoodFacts database, typically prefixed with the language
    code.

    Attributes:
        tax_tag: The taxonomy tag (e.g. en:water, en:breakfasts).
    """

    tax_tag: str

    @field_validator("tax_tag", mode="after")
    @classmethod
    def remove_en_prefix(cls, value: str) -> str:
        """
        Normalizes OFF taxonomy tags by removing the prefix (e.g. en:sugar).
        This is to ensure that the embedder receives only natural language
        terms later on.

        Args:
            value: The taxonomy tag.

        Returns:
            The OFF taxonomy tag without the prefix.
        """
        if ":" in value:
            return value.split(":", 1)[1]
        return value


def get_product_name(entry: Dict) -> str:
    """
    Extraxts the product name.

    Args:
        A dictionary (deserialized JSONL object).

    Returns:
        The product name (e.g. Nutella).
    """
    name = entry.get("product_name_de") or entry.get("product_name") or "Unknown"
    return name.strip()


def get_brand_names(entry: Dict) -> str:
    """
    Extracts the brand name/s.

    Args:
        A dictionary (deserialized JSONL object).

    Returns:
        One ore more brand names (e.g. Ferrero).
    """
    brands = entry.get("brands_hierarchy") or entry.get("brand_tags")

    if brands and isinstance(brands, list):
        brands = [TaxonomyTag(tax_tag=b).tax_tag for b in brands]
        return ", ".join(brands).title()

    brand_old = entry.get("brand_old")
    if brand_old:
        return brand_old.strip().title()

    return "Unknown brand"


def get_ingredient_tags(ingredients: List[Dict[str, Any]]) -> List[str]:
    """
    Recursively extracts all ingredient taxonomy tags from the ingredients list.

    Args:
        ingredients: A list of dictionaries containing metadata about an
                     ingredient. May contain nested lists under the key
                     'ingredients'.

    Returns:
        A flat list of ingredient taxonomy tagtaxonomy tags (e.g. en:sugar, en:water).
    """
    ids = []
    for item in ingredients:
        if "id" in item:
            ids.append(item["id"])

        if "ingredients" in item and isinstance(item["ingredients"], list):
            ids.extend(get_ingredient_tags(item["ingredients"]))
    return ids


def get_ingredients(entry: Dict) -> str | None:
    """
    Extracts the ingredients for a given product.

    Args:
        A dictionary (deserialized JSONL object).

    Returns:
        The ingredients for a given product or None if ingredients are
        missing.
    """
    ingredients = entry.get("ingredients")

    if ingredients:
        tax_tags = get_ingredient_tags(ingredients)
    else:
        return None

    ingredients = [TaxonomyTag(tax_tag=tg).tax_tag for tg in tax_tags]
    ingredients = ", ".join([i for i in ingredients if i])
    return ingredients


def get_categories(entry: Dict) -> str:
    """
    Extract the categories for a given product.

    Args:
        A dictionary (deserialized JSONL object).

    Returns:
        One or more categeories (e.g. sweet-spreads)
    """
    categories = entry.get("categories_tags")
    if categories:
        categories = [TaxonomyTag(tax_tag=c).tax_tag for c in categories]
        return ", ".join(categories)

    categories = entry.get("categories")
    if categories:
        return categories.strip()

    return "Unknown"


def stream_jsonl(file_path: Path) -> Generator[Tuple[Dict[str, Any], int], None, None]:
    """
    Reads a JSON lines file line by line.

    Args:
        The path to the .jsonl file.

    Yields:
        A decoded JSON object as dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        orjson.JSONDecodeError: If a line is not valid JSON.
    """
    try:
        with open(file_path, "rb") as f:
            for line in f:
                length = len(line)
                if line.strip():
                    yield orjson.loads(line), length
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise


def main() -> None:
    file_size = os.path.getsize(DATA_FILE)
    skipped_count = 0
    with (
        open(OUTPUT_FILE, "wb") as f,
        tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Processing Open Food Facts file",
        ) as pbar,
    ):
        for entry, line_length in tqdm(stream_jsonl(DATA_FILE)):
            pbar.update(line_length)
            product_id = entry.get("id")
            if not product_id:
                skipped_count += 1
                continue
            product = {}
            ingredients = get_ingredients(entry)
            if not ingredients:
                skipped_count += 1
                continue
            product_name = get_product_name(entry)
            brand_names = get_brand_names(entry)
            categories = get_categories(entry)

            product["id"] = product_id
            product["product_name"] = product_name
            product["brand_names"] = brand_names
            product["ingredients"] = ingredients
            product["categories"] = categories

            description = (
                f"Product: {product_name}. "
                f"Brands: {brand_names}. "
                f"Ingredients {ingredients}. "
                f"Categories: {categories}."
            )
            product["description"] = description
            f.write(orjson.dumps(product))
            f.write(b"\n")
    print(
        f"Done. {skipped_count} products were skipped because their "
        f"id or ingredients were missing."
    )


if __name__ == "__main__":
    main()
