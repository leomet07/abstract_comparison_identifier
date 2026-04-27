import json
import anthropic
import os
import pandas as pd
from dotenv import load_dotenv
import sys
from tqdm import tqdm

import traceback

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),  # default
)


def generate_prompt(abstract):
    return (
        """Extract any comparisons of biogeochemical properties between between novel waterbodies (any waterbody that is a result of man-made influence, such as but not limited to mining ponds, tailing ponds, agricultural ponds, stormwater ponds, and tailing ponds) and non-novel/natural waterbodies.
The biogeochemistry proprties of interest includes things like: Methane & Greenhouse Gas Emissions, Nitrous Oxide Emissions, Microplastics, Phosphorus Cycling, Pesticides & Agrochemicals, Heavy Metals, Trace Metals, Mercury, and MethylMercury.

Return ONLY JSON: {"comparisons": [{"property": "...", "pond_a": "...", "pond_b": "...", "finding": "..."}]}
If no waterbody-type biogeochemistry comparisons exist, return ONLY JSON {"comparisons": []}. You do not need to provide an explanation if no comparisons exist.\n\n"""
        + f"Abstract: {abstract}"
    )


def extract_comparisons(abstract: str) -> list[dict]:
    generated_prompt = generate_prompt(abstract)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": generated_prompt}],
    )
    text = resp.content[0].text
    # strip markdown fences if present
    text = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    parsed_comparisons = json.loads(text)
    return parsed_comparisons["comparisons"]


def main(results_path, comparisons_output_path):
    results_df = pd.read_csv(results_path)

    all_comparisons = []

    num_of_abstracts_with_at_least_one_comparison = 0
    abstracts_analyzed = 0
    for index, row in tqdm(results_df.iterrows(), total=len(results_df)):
        abstract = row["abstract"]
        doi = row["doi"]
        link = row["link"]
        title = row["title"]
        try:
            comps = extract_comparisons(abstract)
            num_of_abstracts_with_at_least_one_comparison += 1 if len(comps) > 0 else 0
            abstracts_analyzed += 1
            for c in comps:
                c["abstract_idx"] = index
                c["doi"] = doi
                c["title"] = title
                c["link"] = link

            all_comparisons.extend(comps)
            # print(f"Abstract {index}: {len(comps)} comparisons found")
        except Exception as e:
            print(f"Abstract {index}: error - {e}")
            # traceback.print_stack(e)

    percent_abstracts_with_at_least_one_comparison = (
        num_of_abstracts_with_at_least_one_comparison / abstracts_analyzed
    ) * 100

    print(
        f"Percent of Abstracts with at least one comparison: {percent_abstracts_with_at_least_one_comparison:.2f}"
    )

    df = pd.DataFrame(all_comparisons)
    print(df.groupby("property").size())

    df.to_csv(comparisons_output_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])  # input, output
