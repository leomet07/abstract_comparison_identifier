# Identify comparisons from journal article queries

Workflow steps

For steps 1-3, see automated_query_downloader,

1. Assemble query terms, synonyms and all possible permuations
2. Assemble CSV of queries for both WoS and Scopus
3. Search for queries; Download results (title, DOI, abstract)
   This project is for the following steps:
4. Pull comparisons from abstracts
5. Classify comparisons into corresponding categories
6. Run knowledge gap script – see what’s missing, visualize the data, and attach categories back to comparisons
7. Manually go through comparisons using helpful TUI application.

## TODO: Integrate visual explainer for how this tool works.