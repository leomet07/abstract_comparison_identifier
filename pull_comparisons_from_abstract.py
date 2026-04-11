import sys
import os

import pandas as pd
from tqdm import tqdm

print("Loading detect comparisons util...")
import detect_comparisons

print("Loaded detect comparisons util!")


def main(results_path):
    results_df = pd.read_csv(results_path)

    results_df = results_df[~results_df["abstract"].isna()]

    print("About to batch detect comparisons")
    has_comparison_results = detect_comparisons.detect_comparisons_batch(
        results_df["abstract"].to_list()
    )
    has_comparison_results_boolean = list(
        map(lambda result: result["has_comparison"], has_comparison_results)
    )
    results_df["has_comparison"] = has_comparison_results_boolean
    percent_abstracts_with_comparison = results_df["has_comparison"].mean() * 100
    print(
        f"Percent of abstracts with lexical+NLI comparison: {percent_abstracts_with_comparison:.2f}"
    )
    results_df.to_csv(results_df + ".out.complex")


if __name__ == "__main__":
    main(sys.argv[1])
