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
)  # uses ANTHROPIC_API_KEY env var


def generate_prompt(abstract):
    return (
        """Extract any comparisons of biogeochemical properties between between novel water bodies (anything that is a result of man-made influence, such as but not limited to agricultural ponds, stormwater ponds, and tailing ponds) and novel/natural ponds.
Biogeochemistry includes things like: nutrient concentrations (N, P, C, DOC), pH, dissolved oxygen, 
decomposition rates, greenhouse gas fluxes, redox conditions, mineral weathering, etc.

Return ONLY JSON: {"comparisons": [{"property": "...", "pond_a": "...", "pond_b": "...", "finding": "..."}]}
If no pond-type biogeochemistry comparisons exist, return ONLY JSON {"comparisons": []}. You do not need to provide an explanation if no comparisons exist.\n\n"""
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
    print("-------")
    print(text)
    print("-------")
    parsed_comparisons = json.loads(text)
    return parsed_comparisons["comparisons"]


def main(results_path):
    results_df = pd.read_csv(results_path)
    abstracts = results_df["abstract"].to_list()

    all_comparisons = []
    for i, abstract in enumerate(tqdm(abstracts)):
        try:
            comps = extract_comparisons(abstract)
            for c in comps:
                c["abstract_idx"] = i
            all_comparisons.extend(comps)
            print(f"Abstract {i}: {len(comps)} comparisons found")
        except Exception as e:
            print(f"Abstract {i}: error - {e}")
            traceback.print_stack(e)

    # dump results or load into pandas

    print(json.dumps(all_comparisons, indent=2))

    # optional: pandas summary
    df = pd.DataFrame(all_comparisons)
    print(df.groupby("property").size())

    df.to_csv(results_path.removesuffix(".csv") + "_all_comparisons.csv")


if __name__ == "__main__":
    main(sys.argv[1])
