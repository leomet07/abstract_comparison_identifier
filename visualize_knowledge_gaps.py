# 2d matrix is AI-generated plot to analyze the output of pull_comparisons_from_abstract_using_ai.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import json
from pprint import pprint

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

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
# safe to apply visual changes to column names
property_to_rename = {
    "Heavy Metals and Trace Metals": "H. & T. Metals",
    "Synthetic Particles and Contaminants": "Synthetic Particles",
    "Greenhouse Gas Fluxes": "GHG Fluxes",
    "Nitrogen Cycling": "N Cycling",
    "Phosphorus Cycling": "P Cycling",
    "Sulfur Cycling": "S Cycling",
    "Biodiversity & Biological Communities": "Biodiversity",
    "Water Quality (pH, Major Ion concentrations, Dissolved Oxygen, TSS)": "Water Quality",
}
for property in property_to_rename:
    new_name = property_to_rename[property]
    property_categorization[new_name] = property_categorization.pop(property)


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

ax = heatmap_df["Manmade - standing water"].plot.bar(rot=0)
ax.set_ylabel("Number of Unique DOIs")
fig.tight_layout()
fig.savefig(os.path.join(CHART_OUT_DIR, "barplot.png"), dpi=180, bbox_inches="tight")

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 7))

# Use a sequential colormap, but make 0 a distinct "gap" color
cmap = matplotlib.colormaps["YlGnBu"].copy()
cmap.set_under("#f5f0eb")  # warm off-white for zeros

im = ax.imshow(heatmap_df.values, cmap=cmap, vmin=0.5, aspect="auto")

# Annotate cells
for i in range(heatmap_df.shape[0]):
    for j in range(heatmap_df.shape[1]):
        val = heatmap_df.iloc[i, j]
        color = (
            "#aaa"
            if val == 0
            else ("white" if val > heatmap_df.values.max() * 0.6 else "#333")
        )
        text = "—" if val == 0 else str(val)
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold" if val == 0 else "normal",
        )

ax.set_xticks(range(len(heatmap_df.columns)))
ax.set_xticklabels(heatmap_df.columns, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index, fontsize=10)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Unique Papers (DOIs)", fontsize=10)

ax.set_title(
    "Pond Biogeochemistry Literature Coverage\n(— marks knowledge gaps)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Pond / System Type", fontsize=11, labelpad=8)
ax.set_ylabel("Biogeochemical Property", fontsize=11, labelpad=8)

fig.tight_layout()
fig.savefig(os.path.join(CHART_OUT_DIR, "heatmap.png"), dpi=180, bbox_inches="tight")
print("Saved heatmap.png")

# Print gap summary
print("\n--- KNOWLEDGE GAPS (0 papers) ---")
gaps = []
for prop in heatmap_df.index:
    for pond in heatmap_df.columns:
        if heatmap_df.loc[prop, pond] == 0:
            gaps.append((prop, pond))
print(
    f"{len(gaps)} empty cells out of {heatmap_df.size} total ({100*len(gaps)/heatmap_df.size:.0f}%)"
)
for prop, pond in gaps[:20]:
    print(f"  {prop}  x  {pond}")
if len(gaps) > 20:
    print(f"  ... and {len(gaps)-20} more")

plt.show()

plt.show()
