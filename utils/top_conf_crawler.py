"""
Multi-API Academic Paper Crawler  (v2 — journal-reference arXiv queries)
=========================================================================
arXiv is now queried using the `jr:` (journal reference) field, which
returns only papers that have been published in / accepted by a specific
journal or conference — not just preprints.

Other APIs (Semantic Scholar, OpenAlex, Papers With Code) are unchanged.

Fields: Computer Science, Medicine, Chemistry, Biology,
        Materials Science, Physics, Geology, Psychology
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

OUTPUT_DIR = Path("crawled_papers_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many results to fetch per venue query on arXiv (max 100 per request)
MAX_PER_QUERY = 50

# Date range: papers from the last N days (set None to skip date filter)
DAYS_BACK = 365

# ─────────────────────────────────────────────
# arXiv journal-reference search terms per field
# Uses the `jr:` field which matches the paper's
# "journal_ref" metadata — set by authors when
# the paper was accepted/published.
# ─────────────────────────────────────────────

ARXIV_JOURNAL_REFS: dict[str, list[str]] = {
    "Computer Science": [
        "NeurIPS",
        "ICML",
        "ICLR",
        "CVPR",
        "ICCV",
        "ECCV",
        "ACL",
        "EMNLP",
        "NAACL",
        "AAAI",
        "IJCAI",
        "KDD",
        "SIGIR",
        "WSDM",
        "SIGMOD",
        "VLDB",
        "SOSP",
        "OSDI",
        "STOC",
        "FOCS",
        "CCS",
        "ICSE",
        "IEEE Transactions on Pattern Analysis",  # TPAMI
        "IEEE Transactions on Neural Networks",
        "Journal of Machine Learning Research",   # JMLR
        "Transactions on Machine Learning Research",
    ],
    "Medicine": [
        "New England Journal of Medicine",
        "The Lancet",
        "JAMA",
        "British Medical Journal",
        "Nature Medicine",
        "Annals of Internal Medicine",
        "PLOS Medicine",
        "Journal of Clinical Oncology",
        "Circulation",
        "Cell Host",
    ],
    "Chemistry": [
        "Journal of the American Chemical Society",
        "Angewandte Chemie",
        "Nature Chemistry",
        "Chemical Science",
        "ACS Nano",
        "Organic Letters",
        "Journal of Physical Chemistry",
        "Chemistry of Materials",
        "ACS Catalysis",
        "Green Chemistry",
    ],
    "Biology": [
        "Nature",
        "Science",
        "Cell",
        "PLOS Biology",
        "eLife",
        "Nature Methods",
        "Genome Research",
        "Bioinformatics",
        "Nature Biotechnology",
        "Molecular Cell",
        "ISMB",
        "RECOMB",
    ],
    "Materials Science": [
        "Nature Materials",
        "Advanced Materials",
        "Acta Materialia",
        "Physical Review Materials",
        "ACS Applied Materials",
        "npj Computational Materials",
        "Advanced Functional Materials",
        "Materials Today",
        "Carbon",
        "Journal of Materials Science",
    ],
    "Physics": [
        "Physical Review Letters",
        "Physical Review X",
        "Nature Physics",
        "Journal of High Energy Physics",
        "Physical Review D",
        "Communications Physics",
        "SciPost Physics",
        "Physical Review B",
        "Reviews of Modern Physics",
        "Astrophysical Journal",
    ],
    "Geology": [
        "Nature Geoscience",
        "Journal of Geophysical Research",
        "Geology",
        "Earth and Planetary Science Letters",
        "Tectonics",
        "Geophysical Research Letters",
        "Geochimica et Cosmochimica Acta",
        "Journal of Petrology",
        "Lithos",
        "Chemical Geology",
    ],
    "Psychology": [
        "Psychological Science",
        "Journal of Personality and Social Psychology",
        "Cognition",
        "Psychological Review",
        "Nature Human Behaviour",
        "Journal of Experimental Psychology",
        "Psychological Bulletin",
        "Perspectives on Psychological Science",
        "Annual Review of Psychology",
        "CogSci",
    ],
}

# Used by Semantic Scholar and OpenAlex (unchanged from v1)
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


# ─────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────

@dataclass
class Paper:
    title: str
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]           # journal/conference name
    journal_ref: Optional[str]     # raw journal_ref string from arXiv
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
        d["extra"]   = json.dumps(self.extra)
        return d


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_get(url, params=None, headers=None, retries=1, delay=2):
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
# API 1 — arXiv  (journal-reference queries)
# ─────────────────────────────────────────────
# arXiv `jr:` searches the `journal_ref` field
# that authors fill in once their paper is
# accepted. This gives us published papers only.
#
# Query syntax:  jr:"NeurIPS"
# Multiple refs: jr:"NeurIPS" OR jr:"ICML"
# ─────────────────────────────────────────────

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS  = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
    "dc":     "http://purl.org/dc/elements/1.1/",
    "openSearch": "http://a9.com/-/spec/opensearch/1.1/",
}

def crawl_arxiv(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    """
    Query arXiv using journal-reference (jr:) field per venue.
    Groups venues in batches of 5 with OR to reduce total requests.
    """
    papers      : list[Paper] = []
    venues      = ARXIV_JOURNAL_REFS.get(field, [])
    seen_ids    : set[str]    = set()

    if not venues:
        log.info(f"[arXiv] {field}: no journal refs configured, skipping")
        return papers

    # Batch venues into groups of 5 to stay within URL length limits
    batch_size = 5
    batches    = [venues[i:i+batch_size] for i in range(0, len(venues), batch_size)]

    for batch in batches:
        # Build query: jr:"NeurIPS" OR jr:"ICML" ...
        sub_queries = [f'jr:"{v}"' for v in batch]
        search_query = " OR ".join(sub_queries)

        params = {
            "search_query": search_query,
            "start":        0,
            "max_results":  max_results,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        resp = safe_get(ARXIV_API, params=params)
        if not resp:
            log.warning(f"[arXiv] No response for batch: {batch}")
            continue

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            log.warning(f"[arXiv] XML parse error: {e}")
            continue

        for entry in root.findall("atom:entry", ARXIV_NS):
            try:
                # ── IDs
                arxiv_url = None
                arxiv_id  = None
                for link in entry.findall("atom:link", ARXIV_NS):
                    if link.attrib.get("type") == "text/html":
                        arxiv_url = link.attrib.get("href")
                        arxiv_id  = arxiv_url.split("/abs/")[-1] if arxiv_url else None

                # Skip duplicates within this crawl run
                if arxiv_id and arxiv_id.split("v")[0] in seen_ids:
                    continue
                if arxiv_id:
                    seen_ids.add(arxiv_id.split("v")[0])

                # ── Core fields
                title_el = entry.find("atom:title", ARXIV_NS)
                title    = title_el.text.strip().replace("\n", " ") if title_el is not None else ""

                abs_el   = entry.find("atom:summary", ARXIV_NS)
                abstract = abs_el.text.strip() if abs_el is not None else None

                authors  = [
                    a.find("atom:name", ARXIV_NS).text
                    for a in entry.findall("atom:author", ARXIV_NS)
                    if a.find("atom:name", ARXIV_NS) is not None
                ]

                pub_el    = entry.find("atom:published", ARXIV_NS)
                published = pub_el.text if pub_el is not None else None
                year      = int(published[:4]) if published else None

                # ── Journal reference (the key new field)
                jr_el      = entry.find("arxiv:journal_ref", ARXIV_NS)
                journal_ref = jr_el.text.strip() if jr_el is not None else None

                # ── DOI
                doi_el = entry.find("arxiv:doi", ARXIV_NS)
                doi    = doi_el.text.strip() if doi_el is not None else None

                # ── Derive venue from journal_ref, fall back to matched batch entry
                venue = journal_ref or batch[0]

                papers.append(Paper(
                    title       = title,
                    authors     = authors,
                    year        = year,
                    venue       = venue,
                    journal_ref = journal_ref,
                    abstract    = abstract,
                    arxiv_url   = arxiv_url,
                    arxiv_id    = arxiv_id,
                    doi         = doi,
                    field       = field,
                    source_api  = "arXiv",
                ))

            except Exception as e:
                log.debug(f"arXiv entry parse error: {e}")

        # arXiv rate limit: max 1 request per second
        time.sleep(1.5)

    log.info(f"[arXiv] {field}: {len(papers)} published papers (via journal_ref)")
    return papers


# ─────────────────────────────────────────────
# API 2 — Semantic Scholar  (unchanged from v1)
# ─────────────────────────────────────────────

SS_FIELDS = "title,authors,year,venue,externalIds,abstract,openAccessPdf"

def crawl_semantic_scholar(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers = []
    venues  = FIELD_VENUES.get(field, [])
    queries = venues[:5] if venues else [field]

    for query in queries:
        params = {
            "query":  query,
            "fields": SS_FIELDS,
            "limit":  min(max_results, 100),
        }
        resp = safe_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers={"User-Agent": "academic-crawler/2.0"},
        )
        if not resp:
            continue

        for item in resp.json().get("data", []):
            try:
                ext       = item.get("externalIds") or {}
                arxiv_id  = ext.get("ArXiv")
                arxiv_url = arxiv_id_to_url(arxiv_id) if arxiv_id else None
                doi       = ext.get("DOI")
                oa        = item.get("openAccessPdf") or {}

                papers.append(Paper(
                    title       = item.get("title", ""),
                    authors     = [a.get("name", "") for a in (item.get("authors") or [])],
                    year        = item.get("year"),
                    venue       = item.get("venue"),
                    journal_ref = None,
                    abstract    = item.get("abstract"),
                    arxiv_url   = arxiv_url,
                    arxiv_id    = arxiv_id,
                    doi         = doi,
                    field       = field,
                    source_api  = "SemanticScholar",
                    extra       = {
                        "paperId": item.get("paperId"),
                        "pdf_url": oa.get("url"),
                    },
                ))
            except Exception as e:
                log.debug(f"SS parse error: {e}")

        time.sleep(1)

    log.info(f"[SemanticScholar] {field}: {len(papers)} papers")
    return papers


# ─────────────────────────────────────────────
# API 3 — OpenAlex  (unchanged from v1)
# ─────────────────────────────────────────────

OPENALEX_FIELD_MAP = {
    "Computer Science": "computer science",
    "Medicine":         "medicine",
    "Chemistry":        "chemistry",
    "Biology":          "biology",
    "Materials Science":"materials science",
    "Physics":          "physics",
    "Geology":          "geology",
    "Psychology":       "psychology",
}

def crawl_openalex(field: str, max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers     = []
    field_name = OPENALEX_FIELD_MAP.get(field, field)

    date_filter = ""
    if DAYS_BACK:
        since = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
        date_filter = f",from_publication_date:{since}"

    params = {
        "filter":   f"primary_topic.field.display_name:{field_name}{date_filter}",
        "select":   (
            "id,title,authorships,publication_year,primary_location,"
            "abstract_inverted_index,open_access,ids,best_oa_location"
        ),
        "per-page": min(max_results, 200),
        "sort":     "cited_by_count:desc",
        "mailto":   "crawler@example.com",
    }

    resp = safe_get("https://api.openalex.org/works", params=params)
    if not resp:
        return papers

    for item in resp.json().get("results", []):
        try:
            ids     = item.get("ids") or {}
            oa      = item.get("open_access") or {}
            best_oa = item.get("best_oa_location") or {}

            arxiv_id  = None
            arxiv_url = None
            for url_field in [oa.get("oa_url"), best_oa.get("landing_page_url")]:
                if url_field and "arxiv.org" in str(url_field):
                    arxiv_url = url_field
                    arxiv_id  = url_field.split("/abs/")[-1] if "/abs/" in url_field else None
                    break

            doi = (ids.get("doi") or "").replace("https://doi.org/", "") or None

            authors = []
            for a in (item.get("authorships") or []):
                name = (a.get("author") or {}).get("display_name", "")
                if name:
                    authors.append(name)

            primary_loc = item.get("primary_location") or {}
            source      = primary_loc.get("source") or {}
            venue       = source.get("display_name")
            abstract    = _reconstruct_abstract(item.get("abstract_inverted_index"))

            papers.append(Paper(
                title       = item.get("title", ""),
                authors     = authors,
                year        = item.get("publication_year"),
                venue       = venue,
                journal_ref = None,
                abstract    = abstract,
                arxiv_url   = arxiv_url,
                arxiv_id    = arxiv_id,
                doi         = doi,
                field       = field,
                source_api  = "OpenAlex",
                extra       = {"openalex_id": item.get("id")},
            ))
        except Exception as e:
            log.debug(f"OpenAlex parse error: {e}")

    time.sleep(0.5)
    log.info(f"[OpenAlex] {field}: {len(papers)} papers")
    return papers


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    if not inverted_index:
        return None
    try:
        pos_word = [(p, w) for w, ps in inverted_index.items() for p in ps]
        return " ".join(w for _, w in sorted(pos_word))
    except Exception:
        return None


# ─────────────────────────────────────────────
# API 4 — Papers With Code  (CS/ML, unchanged)
# ─────────────────────────────────────────────

def crawl_papers_with_code(max_results: int = MAX_PER_QUERY) -> list[Paper]:
    papers = []
    params = {
        "items_per_page": min(max_results, 50),
        "ordering":       "-arxiv_id",
    }
    resp = safe_get("https://paperswithcode.com/api/v1/papers/", params=params)
    if not resp:
        return papers

    for item in resp.json().get("results", []):
        try:
            arxiv_id  = item.get("arxiv_id")
            arxiv_url = arxiv_id_to_url(arxiv_id) if arxiv_id else None

            papers.append(Paper(
                title       = item.get("title", ""),
                authors     = [],
                year        = int(item.get("published", "0")[:4]) if item.get("published") else None,
                venue       = item.get("proceeding"),
                journal_ref = None,
                abstract    = item.get("abstract"),
                arxiv_url   = arxiv_url,
                arxiv_id    = arxiv_id,
                doi         = None,
                field       = "Computer Science",
                source_api  = "PapersWithCode",
                extra       = {"pwc_id": item.get("id"), "url_pdf": item.get("url_pdf")},
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
    seen_arxiv  : set[str] = set()
    seen_titles : set[str] = set()
    unique = []

    for p in papers:
        if p.arxiv_id:
            aid = p.arxiv_id.split("v")[0]
            if aid in seen_arxiv:
                continue
            seen_arxiv.add(aid)

        norm = p.title.lower().strip()
        if norm in seen_titles:
            continue
        seen_titles.add(norm)

        unique.append(p)

    return unique


# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────

def save_results(papers: list[Paper], output_dir: Path = OUTPUT_DIR):
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    arxiv_papers = [p for p in papers if p.has_arxiv]

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].to_dict().keys()))
            writer.writeheader()
            writer.writerows(r.to_dict() for r in rows)

    # 1. Master CSV
    all_csv = output_dir / f"all_papers_{timestamp}.csv"
    write_csv(all_csv, papers)
    log.info(f"Saved master CSV      → {all_csv}")

    # 2. arXiv-only CSV
    arxiv_csv = output_dir / f"arxiv_papers_{timestamp}.csv"
    write_csv(arxiv_csv, arxiv_papers)
    log.info(f"Saved arXiv-only CSV  → {arxiv_csv}")

    # 3. Per-field JSON
    by_field: dict[str, list] = {}
    for p in papers:
        by_field.setdefault(p.field, []).append(p.to_dict())
    for fn, fp in by_field.items():
        path = output_dir / f"{fn.replace(' ','_').lower()}_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2, ensure_ascii=False)

    # 4. Summary report
    report_path = output_dir / f"summary_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Crawl Summary — {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total papers collected : {len(papers)}\n")
        f.write(f"Papers with arXiv URL  : {len(arxiv_papers)}\n\n")

        f.write("── By Field ──\n")
        for fn, fp in sorted(by_field.items()):
            ax = sum(1 for p in fp if p.get("arxiv_url"))
            f.write(f"  {fn:<25} total={len(fp):>4}  arxiv={ax:>4}\n")

        f.write("\n── By Source API ──\n")
        api_counts: dict[str, int] = {}
        for p in papers:
            api_counts[p.source_api] = api_counts.get(p.source_api, 0) + 1
        for api, cnt in sorted(api_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {api:<25} {cnt}\n")

        f.write("\n── arXiv papers with journal_ref ──\n")
        jr_papers = [p for p in papers if p.source_api == "arXiv" and p.journal_ref]
        f.write(f"  {len(jr_papers)} arXiv papers carry a journal_ref\n")

    log.info(f"Saved summary         → {report_path}")
    return all_csv, arxiv_csv, report_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    all_papers: list[Paper] = []
    fields = list(FIELD_VENUES.keys())

    log.info(f"Starting crawl for {len(fields)} fields …")
    log.info(f"APIs: arXiv (journal_ref), Semantic Scholar, OpenAlex, Papers With Code")
    log.info(f"Date range: last {DAYS_BACK} days\n")

    for field in fields:
        log.info(f"\n{'─'*50}")
        log.info(f"Field: {field}")
        log.info(f"{'─'*50}")

        # arXiv — now searches by journal reference (published papers only)
        all_papers.extend(crawl_arxiv(field))

        # Semantic Scholar — venue-aware keyword search
        # all_papers.extend(crawl_semantic_scholar(field))

        # OpenAlex — best for Medicine, Geology, Psychology
        # all_papers.extend(crawl_openalex(field))

        time.sleep(1)

    # Papers With Code — CS/ML bonus
    # log.info("\n── Papers With Code (CS bonus) ──")
    # all_papers.extend(crawl_papers_with_code())

    log.info(f"\nTotal before dedup : {len(all_papers)}")
    all_papers = deduplicate(all_papers)
    log.info(f"Total after dedup  : {len(all_papers)}")

    all_csv, arxiv_csv, report = save_results(all_papers)

    arxiv_count = sum(1 for p in all_papers if p.has_arxiv)
    print("\n" + "=" * 60)
    print(f"  ✓ Total papers    : {len(all_papers)}")
    print(f"  ✓ With arXiv URL  : {arxiv_count}")
    print(f"  ✓ Output dir      : {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()