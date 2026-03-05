"""
Multi-API Academic Paper Crawler
Fetches papers with arXiv URLs from: arXiv API, Semantic Scholar, OpenAlex, Papers With Code
Fields: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology
"""

import requests
import time
import json
import csv
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

OUTPUT_DIR = Path("crawled_papers")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many results to request per query/field (tune as needed)
MAX_PER_QUERY = 50

# Date range: papers from the last N days (set None to skip date filter)
DAYS_BACK = 365

# Top conferences / venues per field (used for Semantic Scholar / OpenAlex filtering)
FIELD_VENUES = {
    "Computer Science": [
        "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP",
        "NAACL", "SIGIR", "KDD", "WWW", "AAAI", "IJCAI", "ICSE", "FSE",
        "SOSP", "OSDI", "STOC", "FOCS", "CCS", "IEEE S&P", "USENIX Security",
    ],
    "Medicine": [
        "NEJM", "The Lancet", "JAMA", "BMJ", "Nature Medicine",
        "Cell", "PNAS", "Annals of Internal Medicine",
    ],
    "Chemistry": [
        "JACS", "Angewandte Chemie", "Nature Chemistry", "Chemical Science",
        "ACS Nano", "Organic Letters", "Journal of Physical Chemistry",
    ],
    "Biology": [
        "Nature", "Science", "Cell", "PLOS Biology", "eLife",
        "Nature Methods", "Genome Research", "Bioinformatics",
        "ISMB", "RECOMB",
    ],
    "Materials Science": [
        "Nature Materials", "Advanced Materials", "Acta Materialia",
        "Physical Review Materials", "ACS Applied Materials & Interfaces",
        "npj Computational Materials",
    ],
    "Physics": [
        "Physical Review Letters", "Physical Review X", "Nature Physics",
        "Journal of High Energy Physics", "Physical Review D",
        "Communications Physics", "SciPost Physics",
    ],
    "Geology": [
        "Nature Geoscience", "Journal of Geophysical Research", "Geology",
        "Earth and Planetary Science Letters", "Tectonics", "Geophysical Research Letters",
    ],
    "Psychology": [
        "Psychological Science", "Journal of Personality and Social Psychology",
        "Cognition", "Psychological Review", "Nature Human Behaviour",
        "Journal of Experimental Psychology", "CogSci",
    ],
}

# arXiv category codes per field
ARXIV_CATEGORIES = {
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.RO",
        "cs.IR", "cs.CR", "cs.SE", "cs.DS",
    ],
    "Physics": [
        "physics.gen-ph", "hep-th", "hep-ph", "quant-ph",
        "cond-mat.mtrl-sci", "physics.optics", "astro-ph.CO",
    ],
    "Materials Science": ["cond-mat.mtrl-sci", "cond-mat.supr-con", "cond-mat.mes-hall"],
    "Biology": ["q-bio.GN", "q-bio.BM", "q-bio.CB", "q-bio.NC", "q-bio.QM"],
    "Chemistry": ["physics.chem-ph", "q-bio.BM"],
    "Medicine": ["q-bio.TO", "q-bio.GN"],
    "Psychology": ["q-bio.NC"],
    "Geology": ["physics.geo-ph", "astro-ph.EP"],
}


# ─────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────

@dataclass
class Paper:
    title: str
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    arxiv_url: Optional[str]
    arxiv_id: Optional[str]
    doi: Optional[str]
    field: str
    source_api: str
    extra: dict = field(default_factory=dict)

    @property
    def has_arxiv(self) -> bool:
        return bool(self.arxiv_url or self.arxiv_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["authors"] = "; ".join(self.authors)
        d["extra"] = json.dumps(self.extra)
        return d


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_get(url: str, params: dict = None, headers: dict = None, retries=1, delay=2) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 3))
                log.warning(f"Rate limited. Sleeping {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay * (attempt + 1))
    return None


def arxiv_id_to_url(arxiv_id: str) -> str:
    aid = arxiv_id.replace("arXiv:", "").strip()
    return f"https://arxiv.org/abs/{aid}"


# ─────────────────────────────────────────────
# API 1 — arXiv
# ─────────────────────────────────────────────

def crawl_arxiv(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers = []
    cats = ARXIV_CATEGORIES.get(field, [])
    if not cats:
        return papers

    for cat in cats[:3]:  # limit categories per field to avoid overload
        query = f"cat:{cat}"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = safe_get("https://export.arxiv.org/api/query", params=params)
        if not resp:
            continue

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            try:
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.strip()
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                published = entry.find("atom:published", ns).text
                year = int(published[:4]) if published else None

                arxiv_url = None
                arxiv_id = None
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("type") == "text/html":
                        arxiv_url = link.attrib.get("href")
                        arxiv_id = arxiv_url.split("/abs/")[-1] if arxiv_url else None

                # Extract DOI if present
                doi = None
                doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
                if doi_el is not None:
                    doi = doi_el.text

                papers.append(Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=cat,
                    abstract=abstract,
                    arxiv_url=arxiv_url,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    field=field,
                    source_api="arXiv",
                ))
            except Exception as e:
                log.debug(f"arXiv parse error: {e}")

        time.sleep(1.5)  # arXiv rate limit: ≤1 req/sec recommended

    log.info(f"[arXiv] {field}: {len(papers)} papers")
    return papers


# ─────────────────────────────────────────────
# API 2 — Semantic Scholar
# ─────────────────────────────────────────────

SS_FIELDS = "title,authors,year,venue,externalIds,abstract,openAccessPdf"

def crawl_semantic_scholar(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers = []
    venues = FIELD_VENUES.get(field, [])
    queries = venues[:5] if venues else [field]  # use venue names as queries

    for query in queries:
        params = {
            "query": query,
            "fields": SS_FIELDS,
            "limit": min(max_results, 100),
        }
        resp = safe_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers={"User-Agent": "academic-crawler/1.0"},
        )
        if not resp:
            continue

        data = resp.json().get("data", [])
        for item in data:
            try:
                ext = item.get("externalIds", {}) or {}
                arxiv_id = ext.get("ArXiv")
                arxiv_url = arxiv_id_to_url(arxiv_id) if arxiv_id else None
                doi = ext.get("DOI")

                papers.append(Paper(
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    year=item.get("year"),
                    venue=item.get("venue"),
                    abstract=item.get("abstract"),
                    arxiv_url=arxiv_url,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    field=field,
                    source_api="SemanticScholar",
                    extra={"paperId": item.get("paperId")},
                ))
            except Exception as e:
                log.debug(f"SS parse error: {e}")

        time.sleep(1)

    log.info(f"[SemanticScholar] {field}: {len(papers)} papers")
    return papers


# ─────────────────────────────────────────────
# API 3 — OpenAlex
# ─────────────────────────────────────────────

OPENALEX_FIELD_MAP = {
    "Computer Science": "computer science",
    "Medicine": "medicine",
    "Chemistry": "chemistry",
    "Biology": "biology",
    "Materials Science": "materials science",
    "Physics": "physics",
    "Geology": "geology",
    "Psychology": "psychology",
}

def crawl_openalex(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers = []
    field_name = OPENALEX_FIELD_MAP.get(field, field)

    # Build date filter
    date_filter = ""
    if DAYS_BACK:
        since = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
        date_filter = f",from_publication_date:{since}"

    params = {
        "filter": f"primary_topic.field.display_name:{field_name}{date_filter}",
        "select": "id,title,authorships,publication_year,primary_location,abstract_inverted_index,open_access,ids,best_oa_location",
        "per-page": min(max_results, 200),
        "sort": "cited_by_count:desc",
        "mailto": "crawler@example.com",  # OpenAlex politely asks for this
    }

    resp = safe_get("https://api.openalex.org/works", params=params)
    if not resp:
        return papers

    results = resp.json().get("results", [])
    for item in results:
        try:
            ids = item.get("ids", {}) or {}
            arxiv_id = None
            arxiv_url = None

            # Check open_access URL for arXiv links
            oa = item.get("open_access", {}) or {}
            oa_url = oa.get("oa_url", "")
            if oa_url and "arxiv.org" in str(oa_url):
                arxiv_url = oa_url
                arxiv_id = oa_url.split("/abs/")[-1] if "/abs/" in oa_url else None

            # Also check best_oa_location
            best_oa = item.get("best_oa_location") or {}
            landing = best_oa.get("landing_page_url", "")
            if not arxiv_url and "arxiv.org" in str(landing):
                arxiv_url = landing
                arxiv_id = landing.split("/abs/")[-1] if "/abs/" in landing else None

            doi = ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None

            # Authors
            authors = []
            for a in (item.get("authorships") or []):
                author_info = a.get("author") or {}
                name = author_info.get("display_name", "")
                if name:
                    authors.append(name)

            # Venue
            primary_loc = item.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            venue = source.get("display_name")

            # Abstract (OpenAlex stores as inverted index)
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

            papers.append(Paper(
                title=item.get("title", ""),
                authors=authors,
                year=item.get("publication_year"),
                venue=venue,
                abstract=abstract,
                arxiv_url=arxiv_url,
                arxiv_id=arxiv_id,
                doi=doi,
                field=field,
                source_api="OpenAlex",
                extra={"openalex_id": item.get("id")},
            ))
        except Exception as e:
            log.debug(f"OpenAlex parse error: {e}")

    time.sleep(0.5)
    log.info(f"[OpenAlex] {field}: {len(papers)} papers")
    return papers


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    try:
        positions = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions.append((pos, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)
    except Exception:
        return None


# ─────────────────────────────────────────────
# API 4 — Papers With Code (CS / ML focused)
# ─────────────────────────────────────────────

def crawl_papers_with_code(max_results: int = MAX_PER_QUERY) -> list[Paper]:
    """Papers With Code only covers CS/ML but has very high-quality venue metadata."""
    papers = []
    params = {
        "items_per_page": min(max_results, 50),
        "ordering": "-arxiv_id",  # most recently added
    }
    resp = safe_get("https://paperswithcode.com/api/v1/papers/", params=params)
    if not resp:
        return papers

    for item in resp.json().get("results", []):
        try:
            arxiv_id = item.get("arxiv_id")
            arxiv_url = arxiv_id_to_url(arxiv_id) if arxiv_id else None

            papers.append(Paper(
                title=item.get("title", ""),
                authors=[],  # PWC papers endpoint doesn't return authors
                year=int(item.get("published", "0")[:4]) if item.get("published") else None,
                venue=item.get("proceeding"),
                abstract=item.get("abstract"),
                arxiv_url=arxiv_url,
                arxiv_id=arxiv_id,
                doi=None,
                field="Computer Science",
                source_api="PapersWithCode",
                extra={"pwc_id": item.get("id"), "url_pdf": item.get("url_pdf")},
            ))
        except Exception as e:
            log.debug(f"PWC parse error: {e}")

    time.sleep(0.5)
    log.info(f"[PapersWithCode] Computer Science: {len(papers)} papers")
    return papers


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

def deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_arxiv: set[str] = set()
    seen_titles: set[str] = set()
    unique = []

    for p in papers:
        # Deduplicate by arXiv ID first
        if p.arxiv_id:
            aid = p.arxiv_id.split("v")[0]  # strip version suffix
            if aid in seen_arxiv:
                continue
            seen_arxiv.add(aid)

        # Then fall back to normalized title
        norm_title = p.title.lower().strip()
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        unique.append(p)

    return unique


# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────

def save_results(papers: list[Paper], output_dir: Path = OUTPUT_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Master CSV (all papers)
    all_csv = output_dir / f"all_papers_{timestamp}.csv"
    if papers:
        keys = list(papers[0].to_dict().keys())
        with open(all_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(p.to_dict() for p in papers)
        log.info(f"Saved master CSV → {all_csv}")

    # ── 2. arXiv-only CSV
    arxiv_papers = [p for p in papers if p.has_arxiv]
    arxiv_csv = output_dir / f"arxiv_papers_{timestamp}.csv"
    if arxiv_papers:
        keys = list(arxiv_papers[0].to_dict().keys())
        with open(arxiv_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(p.to_dict() for p in arxiv_papers)
        log.info(f"Saved arXiv-only CSV → {arxiv_csv}")

    # ── 3. Per-field JSON
    by_field: dict[str, list] = {}
    for p in papers:
        by_field.setdefault(p.field, []).append(p.to_dict())

    for field_name, field_papers in by_field.items():
        safe_name = field_name.replace(" ", "_").lower()
        json_path = output_dir / f"{safe_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(field_papers, f, indent=2, ensure_ascii=False)

    # ── 4. Summary report
    report_path = output_dir / f"summary_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Crawl Summary — {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total papers collected : {len(papers)}\n")
        f.write(f"Papers with arXiv URL  : {len(arxiv_papers)}\n\n")

        f.write("── By Field ──\n")
        for fn, fp in sorted(by_field.items()):
            arxiv_count = sum(1 for p in fp if p.get("arxiv_url"))
            f.write(f"  {fn:<25} total={len(fp):>4}  arxiv={arxiv_count:>4}\n")

        f.write("\n── By Source API ──\n")
        api_counts: dict[str, int] = {}
        for p in papers:
            api_counts[p.source_api] = api_counts.get(p.source_api, 0) + 1
        for api, cnt in sorted(api_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {api:<25} {cnt}\n")

    log.info(f"Saved summary → {report_path}")
    return all_csv, arxiv_csv, report_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    all_papers: list[Paper] = []
    fields = list(FIELD_VENUES.keys())

    log.info(f"Starting crawl for {len(fields)} fields …")
    log.info(f"APIs: arXiv, Semantic Scholar, OpenAlex, Papers With Code")
    log.info(f"Date range: last {DAYS_BACK} days\n")

    for field in fields:
        log.info(f"\n{'─'*50}")
        log.info(f"Field: {field}")
        log.info(f"{'─'*50}")

        # arXiv (great for CS, Physics, Materials, Biology)
        all_papers.extend(crawl_arxiv(field))

        # Semantic Scholar (broad coverage, venue-aware)
        all_papers.extend(crawl_semantic_scholar(field))

        # OpenAlex (best for Medicine, Geology, Psychology)
        all_papers.extend(crawl_openalex(field))

        time.sleep(1)

    # Papers With Code — CS/ML bonus
    # log.info("\n── Papers With Code (CS bonus) ──")
    # all_papers.extend(crawl_papers_with_code())

    log.info(f"\nTotal before dedup: {len(all_papers)}")
    all_papers = deduplicate(all_papers)
    log.info(f"Total after dedup : {len(all_papers)}")

    all_csv, arxiv_csv, report = save_results(all_papers)

    # Print quick summary to console
    arxiv_count = sum(1 for p in all_papers if p.has_arxiv)
    print("\n" + "=" * 60)
    print(f"  ✓ Total papers    : {len(all_papers)}")
    print(f"  ✓ With arXiv URL  : {arxiv_count}")
    print(f"  ✓ Output dir      : {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()