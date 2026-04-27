import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
from colorama import Fore, Back, Style


def clear():
    print(chr(27) + "[2J")


def read_saves_folder(saves_folder):
    saves = list(os.listdir(saves_folder))
    if len(saves) == 0:
        return None
    saves.sort()
    latest_save_filepath = os.path.join(saves_folder, saves[-1])
    return pd.read_csv(latest_save_filepath)


def display_row(row):
    print(f"Abstract index: {row["abstract_idx"]}")
    print(f"DOI: {row["doi"]}")
    print(f"{row["title"]}")
    print()
    print(
        Fore.LIGHTRED_EX
        + f"{row["pond_a_cat"]} vs {row["pond_b_cat"]}"
        + Style.RESET_ALL
    )
    print(Fore.LIGHTYELLOW_EX + f"Category: {row["prop_cat"]}" + Style.RESET_ALL)
    print(f"Finding: ")
    print(Fore.GREEN + row["finding"] + Style.RESET_ALL)


def main(comparisons_df_with_categories_path, should_read_saves_folder):
    df = pd.read_csv(comparisons_df_with_categories_path)

    saves_folder = os.path.join(
        os.path.dirname(comparisons_df_with_categories_path), "manual_inspection_saves"
    )
    latest_save = None
    if should_read_saves_folder and os.path.exists(saves_folder):
        latest_save = read_saves_folder(saves_folder)

    if not os.path.exists(saves_folder):
        os.makedirs(saves_folder)

    df = df[
        (df["prop_cat"] == "Greenhouse Gas Fluxes")
        | (df["prop_cat"] == "Heavy Metals and Trace Metals")
    ]

    # filter to at least one being manmade
    df = df[
        (df["pond_a_cat"].str.startswith("Manmade"))
        | (df["pond_b_cat"].str.startswith("Manmade"))
    ]

    df = df[
        (
            df["pond_a_cat"].str.startswith("Manmade")
            & df["pond_b_cat"].str.startswith("Natural")
        )
        | (
            df["pond_a_cat"].str.startswith("Natural")
            & df["pond_b_cat"].str.startswith("Manmade")
        )
    ]
    processed_df = (
        [row for index, row in latest_save.iterrows()]
        if latest_save is not None
        else []
    )

    df_left_to_process = (
        pd.concat([latest_save, df]).drop_duplicates(
            subset=["finding", "title", "pond_a_cat", "pond_b_cat", "prop_cat"],
            keep=False,
        )
        if latest_save is not None
        else df
    )
    for index, row in df_left_to_process.iterrows():
        clear()
        display_row(row)

        result = (
            input("Enter (y) if this file contains the proper comparison. ")
            .strip()
            .lower()
        )

        row["inspected"] = True
        row["contains_numerical_flux_comparison"] = result == "y"
        processed_df.append(row)

        # save results
        save_filename = os.path.join(saves_folder, f"{datetime.now()}_save.csv")
        inspected_so_far_df = pd.DataFrame(processed_df)
        inspected_so_far_df.to_csv(save_filename)

        if result.endswith("q"):
            break


if __name__ == "__main__":
    should_read_saves_folder = (
        sys.argv[2].lower() == "true" if len(sys.argv) == 3 else False
    )
    main(sys.argv[1], should_read_saves_folder)
