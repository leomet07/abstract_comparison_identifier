# an AI-generated script to analyze the output of pull_comparisons_from_abstract_using_ai.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import json
from pprint import pprint

input_comparisons = sys.argv[1]
df = pd.read_csv(input_comparisons)
CATEGORIZATION_FOLDER = sys.argv[2]

CHART_OUT_DIR = "charts"
if not os.path.exists(CHART_OUT_DIR):
    os.makedirs(CHART_OUT_DIR)


def get_json_file(filepath):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def pre_process_categories(categorization):
    values_to_categories = {}  # invert the dictionary
    for category, values in categorization.items():
        for value in values:
            values_to_categories[value] = category
    return values_to_categories


pond_categorization = get_json_file(
    os.path.join(CATEGORIZATION_FOLDER, "pond_category_to_ponds.json")
)
property_categorization = get_json_file(
    os.path.join(CATEGORIZATION_FOLDER, "property_category_to_properties.json")
)
# cleaning up column and row names
pond_categorization["Manmade - standing water"] = pond_categorization.pop(
    "Manmade - standing water and impounded natural water bodies"
)
property_categorization["Water Quality"] = property_categorization.pop(
    "Water Quality (pH, Major Ion concentrations, Dissolved Oxygen, TSS)"
)

processed_pond_categorization = pre_process_categories(pond_categorization)
processed_property_categorization = pre_process_categories(property_categorization)


def find_category_for_part(p, procssed_categorization: dict[str]):
    return procssed_categorization.get(p, "Other")


df["prop_cat"] = df["property"].apply(
    lambda p: find_category_for_part(p, processed_property_categorization)
)
df["pond_a_cat"] = df["pond_a"].apply(
    lambda p: find_category_for_part(p, processed_pond_categorization)
)
df["pond_b_cat"] = df["pond_b"].apply(
    lambda p: find_category_for_part(p, processed_pond_categorization)
)

df.to_csv(input_comparisons.replace(".csv", "_with_categories.csv"))

# Melt so each row contributes a comparison for both pond categories
rows = []
for _, r in df.iterrows():
    rows.append(
        {"prop_cat": r["prop_cat"], "pond_cat": r["pond_a_cat"], "doi": r["doi"]}
    )
    rows.append(
        {"prop_cat": r["prop_cat"], "pond_cat": r["pond_b_cat"], "doi": r["doi"]}
    )
melted = pd.DataFrame(rows)

# Pivot: count unique DOIs per cell
pivot = melted.groupby(["prop_cat", "pond_cat"])["doi"].nunique().reset_index()
pivot.columns = ["Property Category", "Pond Category", "n_papers"]
heatmap_df = (
    pivot.pivot(index="Property Category", columns="Pond Category", values="n_papers")
    .fillna(0)
    .astype(int)
)

# Order rows/columns by total (descending)
row_order = heatmap_df.sum(axis=1).sort_values(ascending=False).index
col_order = heatmap_df.sum(axis=0).sort_values(ascending=False).index

# print total by columns
for col in col_order:
    number_of_papers_in_column = heatmap_df[col].sum()
    print(f"{col} has {number_of_papers_in_column} papers")

heatmap_df = heatmap_df.loc[row_order, col_order]

# --- Plot ---
fig = plt.figure(figsize=(14, 7))

ax = heatmap_df["Manmade - standing water"].plot.bar()
ax.set_ylabel("Number of Unique DOIs")
fig.tight_layout()
fig.savefig(os.path.join(CHART_OUT_DIR, "barplot.png"), dpi=180, bbox_inches="tight")

plt.show()
