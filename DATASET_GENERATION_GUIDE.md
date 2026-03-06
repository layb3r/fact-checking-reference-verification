# Citation Dataset Generation Guide

This guide explains how to use the `parse_ref_by_arxivID.py` script to generate a citation dataset from arXiv papers.

## Overview

The script processes arXiv papers and extracts citation instances according to the `dataset_schema.tsx` format. For each citation in a paper, it creates a dataset instance containing:

- **claim_text**: The sentence containing the citation
- **surrounding_context**: Context sentences around  the citation
- **citation_metadata**: Metadata from bib entries (title, authors, venue, year, identifiers)
- **true_outputs**: Ground truth labels (all positive for published papers)

## Prerequisites

Install required dependencies:
```bash
pip install bibtexparser
```

Optionally for LaTeX compilation (if available):
- pdflatex
- bibtex

## Usage

### Basic Usage

```bash
python utils/parse_ref_by_arxivID.py --input path/to/papers.csv
```

### With Options

```bash
python utils/parse_ref_by_arxivID.py \
    --input data/multi-field-papers/arxiv-only-collection/arxiv_papers_20260305_214804.csv \
    --output my_citation_dataset.json \
    --limit 10 \
    --temp-dir temp_downloads \
    --arxiv-col arxiv_id
```

### Command Line Arguments

- `--input` (required): Path to CSV file containing paper list
- `--output` (optional): Output JSON file path (default: auto-generated with timestamp)
- `--limit` (optional): Maximum number of papers to process
- `--temp-dir` (optional): Temporary directory for downloads (default: `temp_processing`)
- `--arxiv-col` (optional): Column name containing arXiv IDs (default: `arxiv_id`)

## Input Format

The input CSV file should contain at least one column with arXiv IDs. Example:

```csv
arxiv_id,title,field
2301.12345,Example Paper Title,Computer Science
2302.67890,Another Paper,Physics
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "creation_date": "2026-03-06T10:30:00",
    "num_papers": 10,
    "num_instances": 250,
    "source": "path/to/input.csv"
  },
  "instances": [
    {
      "claim_text": "Recent work has shown impressive results [CITATION].",
      "surrounding_context": "In the field of machine learning, recent work has shown impressive results [CITATION]. This approach outperforms previous methods.",
      "citation_metadata": {
        "title": "Neural Networks for Everyone",
        "authors": ["John Smith", "Jane Doe"],
        "venue": "NeurIPS",
        "year": 2020,
        "identifiers": {
          "doi": "10.1234/example",
          "arxiv_id": "2020.12345",
          "url": "https://arxiv.org/abs/2020.12345"
        }
      },
      "true_outputs": {
        "true_existence": 1,
        "true_hallucination_category": null,
        "true_alignment": 0,
        "expert_rationale": "Citation from published paper, assumed to be correct and fully supported."
      }
    }
  ]
}
```

## Processing Steps

For each paper, the script:

1. Downloads the LaTeX source from arXiv
2. Extracts the source archive
3. Finds and parses all .bib files
4. Searches .tex files for citation commands (`\cite{key}`, `\citep{key}`, etc.)
5. Extracts claim text and surrounding context for each citation
6. Converts bib entries to metadata format
7. Creates dataset instances with positive ground truth labels

## Ground Truth Labels

For published papers, all instances are labeled as:
- `true_existence`: 1 (reference exists)
- `true_hallucination_category`: null (no hallucination)
- `true_alignment`: 0 (fully supported - assumption)
- `expert_rationale`: "Citation from published paper, assumed to be correct and fully supported."

## Example

```bash
# Process first 5 papers from the arxiv-only collection
python utils/parse_ref_by_arxivID.py \
    --input data/multi-field-papers/arxiv-only-collection/arxiv_papers_20260305_214804.csv \
    --output datasets/positive_citations_sample.json \
    --limit 5
```

## Notes

- **Citation Marker**: All citation commands (`\cite{}`, `\citep{}`, etc.) are replaced with `[CITATION]` in both `claim_text` and `surrounding_context` to clearly indicate where the reference appears
- **Quality Filtering**: Dataset instances are filtered to ensure high quality:
  - Claims must contain the `[CITATION]` marker
  - Claims must be sentence-like (at least 20 characters, 5+ words)
  - LaTeX table/figure commands are filtered out
  - At most 2 instances per reference to avoid redundancy
- **arXiv ID Extraction**: The script extracts arXiv IDs from multiple sources including journal fields like "arXiv preprint arXiv:2009.09761"
- The script automatically cleans up temporary files after processing
- Failed papers are logged but don't stop the overall process
- Citation context extraction includes sentences before and after the citation
- LaTeX commands and comments are cleaned from extracted text
- Multiple citation keys in one command (e.g., `\cite{key1,key2}`) are handled separately
