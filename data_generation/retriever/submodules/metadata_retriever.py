import os
import requests
import json
import re
import logging
from retriever.submodules.reference_schema import ExtractedCitation, Identifiers

# Load email from environment for rate limit handling
EMAIL = os.getenv('EMAIL', '')

# Configure basic logging to catch any API hiccups
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_and_retrieve_citation(citation: ExtractedCitation) -> dict:
    """
    Checks the existence of a paper via Semantic Scholar (primary) and Crossref (fallback).
    Maps the result to the requested JSON schema.
    """
    # 1. Initialize the baseline output schema
    result = {
        "ref_id": citation.ref_id,
        "exists": False,
        "source": None,
        "title": None,
        "year": None,
        "authors": [],
        "identifiers": {
            "doi": None,
            "arxiv_id": None,
            "url": None
        },
        "file_path": None,
        "direct_pdf_url": None  # Deferred for now based on instructions
    }
    
    # 2. Primary Query: Semantic Scholar
    # We merge the title and the first author to query Semantic Scholar[cite: 24].
    title = citation.title or ""
    authors = citation.authors or []
    author_query = authors[0] if authors else ""
    s2_query_string = f"{title} {author_query}".strip()
    
    if s2_query_string:
        try:
            s2_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            s2_params = {
                "query": s2_query_string,
                "limit": 1,
                "fields": "title,year,authors,externalIds,url,openAccessPdf"
            }
            if EMAIL:
                s2_params["api_key"] = EMAIL  # Semantic Scholar uses email for rate limiting
            headers = {"User-Agent": "retriever/1.0 (https://github.com)"}
            response = requests.get(s2_url, params=s2_params, headers=headers, timeout=10)

            if response.ok:
                try:
                    data = response.json().get('data')
                except ValueError:
                    logging.warning("Semantic Scholar returned invalid JSON")
                    data = None

                if data:
                    paper = data[0]
                    ext_ids = paper.get('externalIds', {}) or {}

                    # Populate the schema with any available fields (be forgiving)
                    result["exists"] = True
                    result["source"] = "Semantic Scholar"
                    if paper.get('title'):
                        result["title"] = paper.get('title')
                    if paper.get('year'):
                        result["year"] = paper.get('year')
                    result["authors"] = [a.get('name') for a in paper.get('authors', []) if a.get('name')]

                    # externalIds may have different capitalizations; try common keys
                    doi_val = ext_ids.get('DOI') or ext_ids.get('doi')
                    arxiv_val = ext_ids.get('ArXiv') or ext_ids.get('arXiv') or ext_ids.get('arxiv')
                    result["identifiers"]["doi"] = doi_val
                    result["identifiers"]["arxiv_id"] = arxiv_val
                    if paper.get('url'):
                        result["identifiers"]["url"] = paper.get('url')

                    # Store the direct PDF URL if available
                    oa_pdf = paper.get('openAccessPdf') or {}
                    if isinstance(oa_pdf, dict) and oa_pdf.get('url'):
                        result["direct_pdf_url"] = oa_pdf.get('url')

                    return result
                else:
                    logging.info("Semantic Scholar found no matches")
            else:
                logging.info(f"Semantic Scholar request returned {response.status_code}")

        except requests.RequestException as e:
            logging.warning(f"Semantic Scholar query failed: {e}")

    # 3. Smart Fallback: Crossref
    raw_text = citation.raw_text
    doi = citation.identifiers.doi if citation.identifiers else None
    
    # Try direct DOI lookup first if available
    if doi and not result["exists"]:
        try:
            cr_url = f"https://api.crossref.org/works/{doi}"
            cr_params = {}
            if EMAIL:
                cr_params["mailto"] = EMAIL
            headers = {"User-Agent": "retriever/1.0 (https://github.com)"}
            response = requests.get(cr_url, params=cr_params, headers=headers, timeout=10)

            if response.ok:
                try:
                    data = response.json()
                except ValueError:
                    logging.warning("Crossref DOI lookup returned invalid JSON")
                    data = {}

                if data and data.get('message'):
                    paper = data['message']

                    # Populate the schema
                    result["exists"] = True
                    result["source"] = "Crossref"
                    title_val = paper.get('title') or ['']
                    result["title"] = title_val[0] if isinstance(title_val, list) else title_val
                    result["authors"] = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in paper.get('author', []) if a.get('given') or a.get('family')]
                    result["identifiers"]["doi"] = paper.get('DOI')
                    if paper.get('URL'):
                        result["identifiers"]["url"] = paper.get('URL')

                    return result
                else:
                    logging.info(f"Crossref DOI lookup {doi} found no data")
            else:
                logging.info(f"Crossref DOI lookup returned {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"Crossref DOI lookup failed: {e}")
    
    # Try Crossref when raw_text is available and Semantic Scholar didn't find anything
    if raw_text and not result["exists"]:
        try:
            cr_url = "https://api.crossref.org/works"
            cr_params = {
                "query.bibliographic": raw_text,
                "rows": 1,
                "select": "title,author,DOI,URL"
            }
            if EMAIL:
                cr_params["mailto"] = EMAIL  # Crossref uses mailto for rate limiting
            headers = {"User-Agent": "retriever/1.0 (https://github.com)"}
            response = requests.get(cr_url, params=cr_params, headers=headers, timeout=10)

            if response.ok:
                try:
                    data = response.json()
                except ValueError:
                    logging.warning("Crossref returned invalid JSON")
                    data = {}

                items = data.get('message', {}).get('items') if data else None
                if items:
                    paper = items[0]

                    # Populate the schema
                    result["exists"] = True
                    result["source"] = "Crossref"
                    title_val = paper.get('title') or ['']
                    result["title"] = title_val[0] if isinstance(title_val, list) else title_val
                    result["authors"] = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in paper.get('author', []) if a.get('given') or a.get('family')]
                    result["identifiers"]["doi"] = paper.get('DOI')
                    if paper.get('URL'):
                        result["identifiers"]["url"] = paper.get('URL')

                    return result
                else:
                    logging.info("Crossref found no matches")
            else:
                logging.info(f"Crossref request returned {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"Crossref query failed: {e}")

    # 4. Fallback: Check if a URL is accessible
    # Extract URL from citation or identifiers
    url_to_check = None
    
    # Try URL from identifiers first
    if citation.identifiers and citation.identifiers.url:
        url_to_check = citation.identifiers.url
    
    # Try to extract URL from raw_text if present
    if not url_to_check and citation.raw_text:
        url_match = re.search(r'https?://[^\s\)]+', citation.raw_text)
        if url_match:
            url_to_check = url_match.group(0)
    
    # Check if URL is accessible
    if url_to_check and not result["exists"]:
        try:
            headers = {"User-Agent": "retriever/1.0 (https://github.com)"}
            response = requests.head(url_to_check, headers=headers, timeout=10, allow_redirects=True)
            
            if response.status_code < 400:  # 2xx or 3xx status
                logging.info(f"URL check successful for {citation.ref_id}: {url_to_check} ({response.status_code})")
                result["exists"] = True
                result["source"] = "URL Check"
                result["identifiers"]["url"] = url_to_check
                return result
            else:
                logging.info(f"URL check failed for {url_to_check}: {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"URL check failed for {url_to_check}: {e}")
    
    # Return the default (exists=False) if all methods fail or find nothing
    return result

# --- Example Usage ---
if __name__ == "__main__":
    sample_citation = ExtractedCitation(
        ref_id="R1",
        raw_text="A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in NIPS, 2012.",
        title="Imagenet classification with deep convolutional neural networks",
        authors=["A. Krizhevsky", "I. Sutskever", "G. E. Hinton"],
        venue="NIPS",
        year=2012,
        identifiers=Identifiers(
            doi=None,
            arxiv_id=None,
            url=None
        )
    )

    output = verify_and_retrieve_citation(sample_citation)
    print(json.dumps(output, indent=2))