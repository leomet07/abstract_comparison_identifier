# an AI-generated script to analyze the output of pull_comparisons_from_abstract_using_ai.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os


df = pd.read_csv(sys.argv[1])

CHART_OUT_DIR = "charts"
if not os.path.exists(CHART_OUT_DIR):
    os.makedirs(CHART_OUT_DIR)

## LENNY NOTE: THE TWO FOLLOWING FUNCTIONS ARE A RESULT OF CLAUDE READING AN OUTPUT CSV
## MY TODO: categories need to be updated manually


# --- Categorize properties ---
def categorize_property(p):
    p_lower = p.lower()
    if any(
        k in p_lower
        for k in [
            "methane",
            "ch4",
            "ch₄",
            "greenhouse gas",
            "co2",
            "co₂",
            "carbon dioxide",
            "nitrous oxide",
            "n2o",
            "n₂o",
            "ghg",
            "methane cycling",
            "benthic respiration",
            "ammonia emission",
        ]
    ):
        return "GHG Emissions"
    if any(
        k in p_lower
        for k in [
            "nitrogen",
            "nitrate",
            "nitrite",
            "no3",
            "nh4",
            "ammonium",
            "denitrification",
            "n₂",
            "net n",
            "total kjeldahl",
            "don ",
        ]
    ):
        return "Nitrogen Cycling"
    if any(k in p_lower for k in ["phosphorus", "phosphorous", "tp"]):
        return "Phosphorus Cycling"
    if any(
        k in p_lower
        for k in [
            "heavy metal",
            "trace metal",
            "copper",
            "zinc",
            "lead",
            "cadmium",
            "nickel",
            "chromium",
            "mercury",
            "mehg",
            "methylmercury",
            "arsenic",
            "manganese",
            "iron",
            "selenium",
            "strontium",
            "uranium",
            "aluminum",
            " cu",
            " zn",
            " pb",
            " cd",
            " ni",
            " cr",
            "dissolved metal",
            "particulate copper",
            "particulate zinc",
            "inorganic mercury",
        ]
    ):
        return "Heavy Metals / Trace Metals"
    if any(k in p_lower for k in ["microplastic", "tyre wear", "plastic additive"]):
        return "Microplastics"
    if any(
        k in p_lower
        for k in [
            "pesticide",
            "agrochemical",
            "herbicide",
            "chlorpyrifos",
            "chlordane",
            "terbutylazin",
            "glyphosate",
            "diuron",
            "terbutryn",
        ]
    ):
        return "Pesticides / Agrochemicals"
    if any(k in p_lower for k in ["pah", "polycyclic aromatic"]):
        return "PAHs"
    if any(
        k in p_lower
        for k in [
            "doc",
            "dom ",
            "dissolved organic carbon",
            "dissolved organic matter",
            "carbon burial",
        ]
    ):
        return "DOC / DOM / C Burial"
    if any(
        k in p_lower
        for k in [
            "suspended solid",
            "suspended material",
            "turbidity",
            "tss",
            "sediment retention",
        ]
    ):
        return "TSS / Sediment"
    if any(
        k in p_lower
        for k in [
            "chloride",
            "sodium",
            "conductivity",
            "chlorophyll",
            "coliform",
            "antibiotic",
            "biodiversity",
            "ecotoxicity",
            "eutrophication",
            "water quality",
            "water physicochemical",
            "sulfur",
            "sulfide",
            "irreducible",
            "general pollutant",
        ]
    ):
        return "Other Water Quality"
    return "Other"


# --- Categorize pond types ---
def categorize_pond(p):
    if pd.isna(p):
        return "Other"
    p_lower = p.lower()
    if any(
        k in p_lower
        for k in [
            "natural lake",
            "natural pond",
            "natural waterbod",
            "natural aquatic",
            "natural shallow",
            "natural reference",
            "natural clear",
            "natural dark",
            "reference pond",
            "natural florida",
            "lowland british",
            "natural ponds",
            "typical freshwater",
            "most lakes",
            "lakes, reservoirs",
            "eutrophic lake",
            "small shallow lake",
            "natural urban pond",
            "forest pond",
        ]
    ):
        return "Natural Waterbodies"
    if any(
        k in p_lower
        for k in [
            "natural wetland",
            "naturalized wetland",
            "natural prairie",
            "natural riparian",
            "habitat wetland",
            "marsh, swamp",
            "cattail wetland",
        ]
    ):
        return "Natural Wetlands"
    if any(
        k in p_lower
        for k in [
            "retention pond",
            "retention/detention",
            "stormwater retention",
            "swrp",
            "wet retention",
        ]
    ):
        return "Retention Ponds"
    if any(
        k in p_lower
        for k in [
            "detention pond",
            "detention basin",
            "stormwater detention",
            "dry detention",
            "semi-dry detention",
        ]
    ):
        return "Detention Ponds"
    if any(
        k in p_lower for k in ["wet pond", "wet detention", "wet stormwater", "swdp"]
    ):
        return "Wet Detention Ponds"
    if any(
        k in p_lower
        for k in [
            "stormwater pond",
            "stormwater management",
            "stormwater control",
            "stormwater bmp",
            "stormwater treatment",
            "stormwater wetland",
            "sgi",
            "green infrastructure",
            "stormwater inflow",
            "stormwater runoff",
            "stormwater from",
        ]
    ):
        return "Stormwater Ponds (General)"
    if any(
        k in p_lower
        for k in [
            "constructed wetland",
            "floating treatment",
            "bioretention",
            "bioswale",
            "biofiltration",
            "suds",
            "lid ",
            "low-impact",
            "sand filter",
            "filtration system",
            "sedimentation pond",
            "permeable pavement",
            "grass swale",
            "grass strip",
            "swale",
        ]
    ):
        return "Constructed Treatment / LID"
    if any(
        k in p_lower
        for k in [
            "urban pond",
            "urban artificial",
            "urban park pond",
            "urban ornamental",
            "urban stormwater pond",
            "urban wet",
            "man-made urban",
            "duckweed",
            "chlorinated",
            "non-chlorinated",
            "macrophyte dominated urban",
            "phytoplankton dominated urban",
        ]
    ):
        return "Urban Ponds"
    if any(
        k in p_lower
        for k in [
            "agricultural",
            "rural",
            "farm",
            "feedlot",
            "poultry",
            "aquaculture",
            "fishpond",
            "pastoral",
            "rice cultivation",
        ]
    ):
        return "Agricultural Ponds"
    if any(
        k in p_lower for k in ["highway", "motorway", "road ", "bridge", "expressway"]
    ):
        return "Highway / Road Ponds"
    if any(k in p_lower for k in ["golf course"]):
        return "Golf Course Ponds"
    if any(
        k in p_lower
        for k in [
            "stream",
            "creek",
            "river",
            "tributary",
            "estuary",
            "tidal",
            "coastal",
            "reservoir",
            "impoundment",
        ]
    ):
        return "Streams / Rivers / Reservoirs"
    if any(k in p_lower for k in ["mine", "tailing", "uranium"]):
        return "Mine / Tailing Ponds"
    return "Other"


df["prop_cat"] = df["property"].apply(categorize_property)
df["pond_a_cat"] = df["pond_a"].apply(categorize_pond)
df["pond_b_cat"] = df["pond_b"].apply(categorize_pond)

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
heatmap_df = heatmap_df.loc[row_order, col_order]

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
