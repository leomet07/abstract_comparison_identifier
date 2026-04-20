import json
import anthropic
import os
import pandas as pd
from dotenv import load_dotenv
import sys
from tqdm import tqdm
import numpy as np
from pprint import pprint

import traceback

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),  # default
)


def categorize_properties(properties: list[str]) -> dict:
    """
    Categorize biogeochemical properties into broad groups using Claude Haiku.

    Args:
        properties: List of property name strings to categorize.
        max_categories: Upper bound on number of categories (default 10).

    Returns:
        Dict mapping category name -> list of properties assigned to it.
    """

    prompt = f"""You are sorting a list of biogeochemical properties into seven broad, mutually exclusive categories that I will specify.

Rules:
- Match each proprty to only ONE of the biogeochemistry groupings.
- Every property in the input list must appear in exactly one category.
- Do not rename, rewrap, or paraphrase properties — copy each string verbatim.
- Return ONLY a valid JSON object. No prose, no markdown fences, no commentary.
- If something really cannot be classified, it may go into an "Other" category - but this should be for extreme outiers and edge-cases only.
- Schema: {{"category name": ["property 1", "property 2", ...], ...}}

Here are the categories of biogeochemichal properties (7 properties):
Greenhouse Gas Fluxes
Carbon Cycling
Nitrogen Cycling
Phosphorus Cycling
Heavy Metals and Trace Metals
Organic Pollutants
Synthetic Particles and Contaminants

Input list ({len(properties)} items):
{json.dumps(properties, ensure_ascii=False, indent=2)}
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip accidental code fences just in case
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")

    return json.loads(text)


def categorize_water_bodies(water_bodies: list[str], max_categories: int = 10) -> dict:
    """
    Categorize water body types into broad groups using Claude Haiku,
    with emphasis on distinguishing manmade (novel) vs. natural systems.
    """

    prompt = f"""You are sorting a list of water body types into fewer than {max_categories} broad, mutually exclusive categories.

The primary organizing principle is origin: manmade (novel, engineered, constructed) water bodies vs. natural water bodies. Within each of those two top-level groups, create sub-categories based on hydrologic form.

Rules:
- Every water body type in the input list must appear in exactly one category.
- Do not rename, rewrap, or paraphrase entries — copy each string verbatim.
- Manmade / novel systems include things like: reservoirs, canals, ditches, stormwater ponds, retention/detention basins, wastewater lagoons, constructed wetlands, rice paddies, aquaculture ponds, mine pits, borrow pits, impoundments, and artificial lakes.
- Natural systems include things like: rivers, streams, creeks, lakes, ponds (natural), wetlands (marshes, swamps, bogs, fens, peatlands), estuaries, lagoons (coastal), springs, oxbows, and floodplains.
- If an entry is ambiguous (e.g. "pond" with no qualifier), place it in a clearly labeled "ambiguous / unspecified" category rather than guessing.
- Category names should make the manmade vs. natural distinction obvious (e.g. "Manmade — standing water", "Natural — wetlands", "Natural — lotic systems").
- Return ONLY a valid JSON object. No prose, no markdown fences, no commentary.
- Schema: {{"category name": ["type 1", "type 2", ...], ...}}

Input list ({len(water_bodies)} items):
{json.dumps(water_bodies, ensure_ascii=False, indent=2)}
"""

    # Use the streaming context manager — accumulates the full response
    # while keeping the connection alive for long jobs.
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=32000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        text = "".join(chunk for chunk in stream.text_stream)

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")

    return json.loads(text)


def main(comparisons_output_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(comparisons_output_path)
    print(df)
    unique_properties = df["property"].unique()
    unique_pond_nouns = pd.concat(
        [df["pond_a"], df["pond_b"]], ignore_index=True
    ).unique()

    print("# of unique proprties: ", len(unique_properties))
    print("# of unique pond nouns: ", len(unique_pond_nouns))

    print("About to categorize properties.")
    property_category_to_properties = categorize_properties(unique_properties.tolist())
    print("Categorized proprties!")
    print(property_category_to_properties.keys())
    with open(
        os.path.join(output_dir, "property_category_to_properties.json"), "w"
    ) as file:
        file.write(json.dumps(property_category_to_properties, indent=2))

    print("About to categorize pond nouns.")
    pond_category_to_ponds = categorize_water_bodies(unique_pond_nouns.tolist())
    print("Categorized  pond nouns!")
    print(pond_category_to_ponds.keys())

    with open(os.path.join(output_dir, "pond_category_to_ponds.json"), "w") as file:
        file.write(json.dumps(pond_category_to_ponds, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])  # the comparisons df, output
