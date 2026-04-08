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
    results_df.to_csv(results_df + ".out")


if __name__ == "__main__":
    main(sys.argv[1])
