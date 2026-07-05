import os
import json

EMAIL = os.getenv("EMAIL")
ALLOW_SCIHUB = os.getenv("ALLOW_SCIHUB", "False").lower() == "true"

from retriever.submodules.arxiv_retriever import download_pdf
from pypaperretriever.paper_retriever import PaperRetriever 
from retriever.submodules.reference_schema import ExtractedCitation, Identifiers
from retriever.submodules.metadata_retriever import verify_and_retrieve_citation

def retrieve_paper_arxiv(arxiv_id: str, directory: str, filename = None):
    """Attempt download from arXiv and return a result dict matching schema."""
    ref_id = filename if filename is not None else arxiv_id
    base = {
        "ref_id": ref_id,
        "exists": False,
        "source": None,
        "title": None,
        "year": None,
        "authors": [],
        "identifiers": {
            "doi": None,
            "arxiv_id": arxiv_id,
            "url": None,
        },
        "file_path": None,
    }
    try:
        path = download_pdf(arxiv_id, directory, filename)
        if path:
            base["exists"] = True
            base["source"] = "arXiv"
            base["file_path"] = path
        return base
    except Exception:
        base["exists"] = False
        base["source"] = None
        return base

def retrieve_paper_doi(doi: str, directory: str, filename = None):
    ref_id = filename
    base = {
        "ref_id": ref_id,
        "exists": False,
        "source": None,
        "title": None,
        "year": None,
        "authors": [],
        "identifiers": {
            "doi": doi,
            "arxiv_id": None,
            "url": None,
        },
        "file_path": None,
    }
    retriever = PaperRetriever(
        email=EMAIL,
        doi=doi,
        allow_scihub=ALLOW_SCIHUB,
        download_directory=directory,
        filename=filename,
    )
    try:
        retriever.download()
    except Exception:
        return base

    #Check if file was downloaded
    if retriever.is_downloaded and retriever.filepath and os.path.exists(retriever.filepath):
        base["exists"] = True
        base["source"] = "DOI"
        base["file_path"] = retriever.filepath
    # If we couldn't download but the DOI/Crossref/Unpaywall/Sci-Hub check indicated the
    # paper exists, mark it as existing but leave `file_path` as None (inaccessible).
    elif getattr(retriever, 'found', False):
        base["exists"] = True
        base["source"] = "DOI"
    return base
    

def retrieve_paper(Citation: ExtractedCitation, directory: str = "./down"):
    """Attempt to retrieve paper via arXiv, DOI (optionally Sci-Hub), or metadata.

    Always returns a dict matching the requested JSON schema.
    """
    # Baseline schema populated from the provided Citation
    result = {
        "ref_id": Citation.ref_id,
        "exists": False,
        "source": None,
        "title": Citation.title,
        "year": Citation.year,
        "authors": Citation.authors or [],
        "identifiers": {
            "doi": Citation.identifiers.doi,
            "arxiv_id": Citation.identifiers.arxiv_id,
            "url": Citation.identifiers.url,
        },
        "file_path": None,
    }

    # Track which identifiers have already been tried to avoid duplicates
    tried_arxiv_ids = set()
    tried_dois = set()

    # 1) Try arXiv identifier first if present
    if Citation.identifiers.arxiv_id:
        tried_arxiv_ids.add(Citation.identifiers.arxiv_id)
        arxiv_res = retrieve_paper_arxiv(Citation.identifiers.arxiv_id, directory, filename=Citation.ref_id)
        if arxiv_res and arxiv_res.get("exists"):
            # merge available metadata
            result.update({k: v for k, v in arxiv_res.items() if k in ("exists", "source", "file_path")})
            result["identifiers"]["arxiv_id"] = arxiv_res["identifiers"].get("arxiv_id")
            return result

    # 2) Try DOI retrieval if allowed
    if Citation.identifiers.doi and ALLOW_SCIHUB:
        tried_dois.add(Citation.identifiers.doi)
        doi_res = retrieve_paper_doi(Citation.identifiers.doi, directory, filename=Citation.ref_id)
        if doi_res and doi_res.get("exists"):
            result.update({k: v for k, v in doi_res.items() if k in ("exists", "source", "file_path")})
            result["identifiers"]["doi"] = doi_res["identifiers"].get("doi")
            return result

    # 3) Use metadata lookup to find identifiers, then retry retrievals
    o = verify_and_retrieve_citation(Citation)
    # Merge any metadata information returned
    if o:
        # Merge fields where available
        for key in ("title", "year", "authors"):
            if o.get(key):
                result[key] = o.get(key)
        # If metadata lookup found the paper (via URL check or APIs), mark it as existing
        if o.get("exists"):
            result["exists"] = True
            result["source"] = o.get("source")
            if o.get("identifiers", {}).get("url"):
                result["identifiers"]["url"] = o.get("identifiers", {}).get("url")
        ids = o.get("identifiers", {})
        # Only try arXiv if we haven't already tried this specific ID
        if ids.get("arxiv_id") and ids.get("arxiv_id") not in tried_arxiv_ids:
            arxiv_res = retrieve_paper_arxiv(ids.get("arxiv_id"), directory, filename=Citation.ref_id)
            if arxiv_res and arxiv_res.get("exists"):
                result.update({k: v for k, v in arxiv_res.items() if k in ("exists", "source", "file_path")})
                result["identifiers"]["arxiv_id"] = ids.get("arxiv_id")
                return result
        # Only try DOI if we haven't already tried this specific ID
        if ids.get("doi") and ids.get("doi") not in tried_dois and ALLOW_SCIHUB:
            doi_res = retrieve_paper_doi(ids.get("doi"), directory, filename=Citation.ref_id)
            if doi_res and doi_res.get("exists"):
                result.update({k: v for k, v in doi_res.items() if k in ("exists", "source", "file_path")})
                result["identifiers"]["doi"] = ids.get("doi")
                return result

    # Nothing found/downloaded; return schema indicating not found
    return result

def retrieve_paper_by_metadata(Citation: ExtractedCitation, directory: str = "./down"):
    return verify_and_retrieve_citation(Citation)


def retrieve_papers_batch(input_json_path: str, output_json_path: str, directory: str = "./down"):
    """Process a list of citation dicts from `input_json_path`, retrieve each paper,
    and write a JSON list of results to `output_json_path`.

    The input should be a JSON file containing a list of objects with fields
    compatible with `ExtractedCitation` (including an `identifiers` map).
    Returns the list of result dicts.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    results = []
    for item in items:
        ids = item.get('identifiers', {}) or {}
        identifiers = Identifiers(
            doi=ids.get('doi'),
            arxiv_id=ids.get('arxiv_id'),
            url=ids.get('url'),
        )
        citation = ExtractedCitation(
            ref_id=item.get('ref_id'),
            raw_text=item.get('raw_text'),
            title=item.get('title'),
            authors=item.get('authors', []) or [],
            venue=item.get('venue'),
            year=item.get('year'),
            identifiers=identifiers,
        )
        res = retrieve_paper(citation, directory)
        results.append(res)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results
if __name__ == "__main__":
    # Example usage
    print(f"Email: {EMAIL}")
    print(f"Allow SciHub: {ALLOW_SCIHUB}")
    citation = ExtractedCitation(
        ref_id="ref1",
        raw_text=None,
        title=None,
        authors=[],
        venue=None,
        year=None,
        identifiers=Identifiers(
            doi="10.1007/978-1-62703-535-4_20",
            arxiv_id=None,
            url=None
        )
    )

    result = retrieve_paper(citation, directory="./down")
    # Print JSON to stdout for downstream consumers
    print(json.dumps(result, indent=2))