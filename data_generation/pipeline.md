- Use `top_conf_crawler.py` to get papers from top conferences on different fields (CS, Physiscs, Biology, etc)

- Use a multi-field-sweeper to sweep first-k papers for each field to reduce number of papers to a specified number

- Use `parse_ref_by_arxivID.py` to generate dataset instances from arXiv source:
  - **Input**: CSV with arXiv IDs (from `top_conf_crawler.py`)
  - **Step 1**: Download source from arXiv (`arxiv.org/src/{id}` → tar.gz)
  - **Step 2**: Extract source archive
  - **Step 3**: Parse `.bib` files → `references` dict (cite_key → bib_entry)
  - **Step 4**: Scan `.tex` files for `\cite{...}` commands → `citations` dict (cite_key → list of citation locations)
  - **Step 5 (Backward mapping, default `--mode backward`)**: For each cite_key with a bib_entry:
    - Extract the claim sentence containing the citation from surrounding paragraph
    - Validate claim (≥20 chars, has `[CITATION]` marker, no LaTeX table/figure commands, ≥5 words)
    - Create dataset instance with `claim_text`, `surrounding_context`, `citation_metadata`, `true_outputs`
    - Cap: at most **2 instances per reference**
  - **Alternative (Forward mapping, `--mode forward`)**: First scan all `.tex` files to collect up to **50 valid claims per paper**, then match each claim to its reference(s)
  - **Output**: JSON with `{metadata, instances[]}` where each instance has:
    - `claim_text`: Sentence with `[CITATION]` marker (LaTeX-encoded)
    - `surrounding_context`: Paragraph with `[CITATION]` marker (LaTeX-encoded)
    - `citation_metadata`: `{title, authors[], venue, year, identifiers: {doi, arxiv_id, url}}`
    - `true_outputs`: `{true_existence: 1, true_hallucination_category: null, true_alignment: 0, expert_rationale}`
  - **Usage**: `python parse_ref_by_arxivID.py --input papers.csv --output dataset.json [--mode forward] [--limit N]`

- Postprocess the generated JSON with `postprocess.py`:
  - **Step 1**: `postprocess_missing_field(in_dir, failed_dir, out_dir)` — remove instances with null/empty `title`, `venue`, `year` or empty `authors`; save removed to `failed_dir`
  - **Step 2**: `postprocess_latex()` — convert LaTeX encoding (`\textit{}`, `\'e`, etc.) to plain text in all text fields using `pylatexenc.LatexNodes2Text`
  - **Step 3**: `clean_some_arxiv_and_too_long_authors()` — remove instances where venue contains "arxiv" (case-insensitive) or author count ≥ 10
  - **Step 4**: `check_claim_text(in_dir, out_dir)` — keep only instances where `claim_text` appears within `surrounding_context`
  - **Combined**: `combined_postprocess(in_dir, failed_dir, out_dir)` — runs all four steps sequentially in a single pass

Instance format:

```
{
    "original_paper": "2603.03973v1",
    "claim_text": "...",
    "surrounding_context": "...",
    "citation_metadata": {
        "title": "...",
        "authors": [
            "Karras, Tero",
            "Laine, Samuli",
            "Lehtinen, Jaakko",
            "Aila, Timo"
        ],
        "venue": "...",
        "year": 2019,
        "identifiers": {
            "doi": null,
            "arxiv_id": null,
            "url": null
        }
    },
    "true_outputs": {
        "true_existence": 1,
        "true_hallucination_category": null,
        "true_alignment": 0,
        "expert_rationale": "Citation from published paper, assumed to be correct and fully supported."
    }
}
```