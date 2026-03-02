"""
GROBID-based reference extraction and normalization utilities.

Code này được tách ra từ `extract_references_type.py` để dùng như
module trong package `extract_references_type`, phù hợp với pipeline.
"""

import os
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


def _safe_text(el) -> Optional[str]:
    if not el:
        return None
    t = el.get_text(strip=True)
    return t if t else None


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class GROBIDReferenceExtractor:
    """Class để trích xuất references từ PDF sử dụng GROBID"""

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        Khởi tạo extractor

        Args:
            grobid_url: URL của GROBID server (mặc định: http://localhost:8070)
        """
        self.grobid_url = grobid_url.rstrip("/")
        self.api_endpoint = f"{self.grobid_url}/api/processReferences"

    def check_server(self) -> bool:
        """Kiểm tra xem GROBID server có đang chạy không"""
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def extract_references_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Trích xuất references từ file PDF

        Args:
            pdf_path: Đường dẫn đến file PDF

        Returns:
            List các raw text references
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")

        # Kiểm tra server
        if not self.check_server():
            raise ConnectionError(
                f"Không thể kết nối đến GROBID server tại {self.grobid_url}. "
                "Hãy đảm bảo GROBID server đang chạy."
            )

        # Gửi file PDF đến GROBID
        with open(pdf_path, "rb") as pdf_file:
            files = {"input": pdf_file}
            data = {
                "consolidateCitations": "0",
                "includeRawCitations": "1",
                "includeRawAffiliations": "0",
            }

            try:
                response = requests.post(
                    self.api_endpoint,
                    files=files,
                    data=data,
                    timeout=120,
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise Exception(f"Lỗi khi gửi request đến GROBID: {e}")

        # Parse XML response
        xml_content = response.text
        return self._parse_references(xml_content)

    def _parse_references(self, xml_content: str) -> List[str]:
        """
        Parse XML từ GROBID để lấy danh sách raw text references
        """
        soup = BeautifulSoup(xml_content, "xml")
        references: List[str] = []

        # Tìm tất cả các thẻ <biblStruct>
        bibl_structs = soup.find_all("biblStruct")

        for bibl in bibl_structs:
            note = bibl.find("note", type="raw_reference")
            if note:
                raw_text = note.get_text(strip=True)
                if raw_text:
                    references.append(raw_text)

        # Nếu không tìm thấy biblStruct, thử tìm trong note
        if not references:
            notes = soup.find_all("note", type="biblio")
            for note in notes:
                raw_text = note.get_text(strip=True)
                if raw_text:
                    references.append(raw_text)

        return references

    def split_merged_references(self, references: List[str]) -> List[str]:
        """
        Tách các raw references bị GROBID gộp lại thành 1 entry.
        Detect bằng cách tìm pattern: có >= 2 title trong ngoặc kép,
        mỗi title đi kèm với tác giả phía trước.
        """
        split_results: List[str] = []

        for raw in references:
            parts = re.split(
                r'(?<=\.)\s+(?=[A-Z]\.\s+[A-Z][a-z].*?[,]\s*(?:[A-Z]\.\s*)*[A-Z][a-z].*?[""\u201c])',
                raw,
            )

            if len(parts) > 1:
                split_results.extend(parts)
            else:
                split_results.append(raw)

        return split_results

    def parse_citation(self, raw_text: str) -> Dict:
        """
        Parse 1 citation qua /api/processCitation, trả về flat dict.
        """
        endpoint = f"{self.grobid_url}/api/processCitation"

        try:
            response = requests.post(
                endpoint,
                data={"citations": raw_text, "consolidateCitations": "0"},
                headers={"Accept": "application/xml"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException:
            return {"raw_text": raw_text}

        soup = BeautifulSoup(response.text, "xml")
        result: Dict[str, Any] = {"raw_text": raw_text}

        # ---- Title ----
        title_a = soup.find("title", level="a")  # bài báo/chương
        title_m = soup.find("title", level="m")  # sách/proceedings
        if title_a:
            result["title"] = _safe_text(title_a)
            if title_m:
                result["proceedings_title"] = _safe_text(title_m)
        else:
            if title_m:
                result["title"] = _safe_text(title_m)

        # ---- Authors (structured from TEI) ----
        authors: List[Dict[str, str]] = []
        for author in soup.find_all("author"):
            pers = author.find("persName")
            if pers:
                given_parts: List[str] = []
                for forename in pers.find_all("forename"):
                    txt = _safe_text(forename)
                    if txt:
                        given_parts.append(txt)
                surname = _safe_text(pers.find("surname"))

                given_name = " ".join(given_parts) if given_parts else None
                family_name = surname if surname else None

                if given_name or family_name:
                    authors.append(
                        {
                            "given_name": given_name or "",
                            "family_name": family_name or "",
                        }
                    )
                    continue

            org = author.find("orgName")
            if org:
                org_name = _safe_text(org)
                if org_name:
                    authors.append({"given_name": org_name, "family_name": ""})

        if authors:
            result["authors_structured"] = authors

        # ---- Year ----
        date_elem = soup.find("date")
        if date_elem:
            when = date_elem.get("when")
            if when:
                result["year"] = when

        # ---- Journal / Venue ----
        journal = soup.find("title", level="j")
        if journal:
            result["journal"] = _safe_text(journal)

        # ---- Volume / Issue / Pages ----
        vol = soup.find("biblScope", unit="volume")
        if vol:
            result["volume"] = _safe_text(vol)

        issue = soup.find("biblScope", unit="issue")
        if issue:
            result["issue"] = _safe_text(issue)

        pages = soup.find("biblScope", unit="page")
        if pages:
            f = pages.get("from", "")
            t = pages.get("to", "")
            if f and t:
                result["pages"] = f"{f}-{t}"
            else:
                result["pages"] = _safe_text(pages)

        # ---- DOI / ISBN / URL / Report number ----
        for idno in soup.find_all("idno"):
            t = str(idno.get("type") or "").strip().lower()
            val = _safe_text(idno)
            if not val:
                continue
            if t == "doi":
                result["doi"] = val
            elif "isbn" in t:
                result["isbn"] = val
            elif t in ("url", "uri", "link"):
                result["url"] = val
            elif "report" in t:
                result["report_number"] = val

        ptr = soup.find("ptr")
        if not result.get("url") and ptr and ptr.get("target"):
            result["url"] = str(ptr["target"]).strip()

        ref = soup.find("ref", target=True)
        if not result.get("url") and ref and ref.get("target"):
            result["url"] = str(ref["target"]).strip()

        pub = soup.find("publisher")
        if pub:
            result["publisher"] = _safe_text(pub)

        for tag in ["pubPlace", "address"]:
            loc = soup.find(tag)
            if loc:
                loc_text = _safe_text(loc)
                if loc_text:
                    if "," in loc_text:
                        parts = [p.strip() for p in loc_text.split(",", 1)]
                        city_part = parts[0]
                        pub_part = parts[1] if len(parts) > 1 else ""
                        if pub_part and re.search(
                            r"\bpress\b|\bpublish\b|\barnold\b|\bwiley\b"
                            r"|\belsevier\b|\bspringer\b|\bwilkins\b|\bbooks?\b",
                            pub_part,
                            re.IGNORECASE,
                        ):
                            result["location"] = city_part
                            if not result.get("publisher"):
                                result["publisher"] = pub_part
                        else:
                            result["location"] = loc_text
                    else:
                        result["location"] = loc_text
                break

        meeting = soup.find("meeting")
        if "location" not in result and meeting:
            addr = meeting.find("address")
            if addr:
                parts: List[str] = []
                for sub in ["settlement", "region", "country"]:
                    el = addr.find(sub)
                    t = _safe_text(el) if el else None
                    if t:
                        parts.append(t)
                if parts:
                    result["location"] = ", ".join(parts)

        return result

    def fill_missing_from_raw(self, parsed: Dict) -> Dict:
        """
        Bổ sung thông tin còn thiếu từ raw_text bằng regex.
        """
        raw = parsed.get("raw_text", "") or ""
        if not raw:
            return parsed

        if "year" not in parsed:
            m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", raw)
            if m:
                parsed["year"] = m.group(1)
                parsed["year_source"] = "regex"

        if "title" not in parsed:
            title_match = re.search(r'["\u201c](.+?)["\u201d]', raw)
            if title_match:
                parsed["title"] = title_match.group(1)
                parsed["title_source"] = "regex"
            else:
                sentences = re.split(r"(?<![A-Z])\.\s+", raw.rstrip("."))
                title_found = False
                if len(sentences) >= 2:
                    for i in range(1, len(sentences)):
                        candidate = sentences[i].strip()
                        if candidate and len(candidate) > 5 and not re.match(
                            r"^(arXiv|CoRR|In\s|pp\.|vol\.|http|doi:|Available|Accessed|\d{4})",
                            candidate,
                            re.IGNORECASE,
                        ):
                            parsed["title"] = candidate
                            parsed["title_source"] = "regex"
                            title_found = True
                            break

                if not title_found:
                    book_match = re.search(
                        r"(?:,\s*)([A-Z][^,]{5,}?)(?:,\s*(?:[A-Z][a-z]+,|vol\.|pp\.|[12]\d{3}))",
                        raw,
                    )
                    if book_match:
                        candidate = book_match.group(1).strip()
                        if len(candidate) > 5 and not re.match(r"^[A-Z]\.\s", candidate):
                            parsed["title"] = candidate
                            parsed["title_source"] = "regex"

        if "doi" not in parsed:
            m = re.search(r"(10\.\d{4,}/[^\s,;]+)", raw)
            if m:
                parsed["doi"] = m.group(1).rstrip(".")
                parsed["doi_source"] = "regex"

        if "isbn" not in parsed:
            m = re.search(r"\bISBN:\s*([0-9Xx-]{10,20})\b", raw)
            if m:
                parsed["isbn"] = m.group(1)
                parsed["isbn_source"] = "regex"

        if "url" not in parsed:
            m = re.search(r"(https?://[^\s\)]+)", raw)
            if m:
                parsed["url"] = m.group(1).rstrip(".")
                parsed["url_source"] = "regex"

        if "access_date" not in parsed:
            m = re.search(
                r"\b(?:Accessed|Retrieved)\s+(\d{4}-\d{2}-\d{2})\b",
                raw,
                re.IGNORECASE,
            )
            if m:
                parsed["access_date"] = m.group(1)
                parsed["access_date_source"] = "regex"

        if "arxiv_id" not in parsed:
            m = re.search(
                r"(?:arXiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(v\d+)?",
                raw,
                re.IGNORECASE,
            )
            if m:
                parsed["arxiv_id"] = m.group(1)
                if m.group(2):
                    parsed["version"] = m.group(2)

        if "issue" not in parsed or "volume" not in parsed:
            m = re.search(r"\b(\d+)\s*\(\s*(\d+)\s*\)", raw)
            if m:
                parsed.setdefault("volume", m.group(1))
                parsed.setdefault("issue", m.group(2))

        if "pages" not in parsed and "pages_or_article_number" not in parsed:
            m = re.search(
                r"\b(?:pp\.?|p\.)\s*(\d+)\s*[-–]\s*(\d+)\b", raw, re.IGNORECASE
            )
            if m:
                parsed["pages"] = f"{m.group(1)}-{m.group(2)}"
                parsed["pages_source"] = "regex"
            else:
                m2 = re.search(r"[:]\s*([eE]?\d{3,})\b", raw)
                if m2:
                    parsed["pages_or_article_number"] = m2.group(1)

        if "report_number" not in parsed:
            m = re.search(
                r"\bReport\s+No:\s*([A-Za-z0-9\.-]+)\b", raw, re.IGNORECASE
            )
            if m:
                parsed["report_number"] = m.group(1)

        if "edition" not in parsed:
            m = re.search(r"\b(\d+(?:st|nd|rd|th))\s*ed\b", raw, re.IGNORECASE)
            if m:
                parsed["edition"] = m.group(1)

        if "degree_type" not in parsed:
            m = re.search(r"\b(PhD|MD|MSc|MS|MA|MBA|BSc|BS)\b", raw)
            if m:
                parsed["degree_type"] = m.group(1)

        if "institution" not in parsed:
            m = re.search(
                r"\.\s*([^\.]*(University|Institute|College|Hospital)[^\.]*)\.\s*(?:PhD|dissertation|thesis)",
                raw,
                re.IGNORECASE,
            )
            if m:
                parsed["institution"] = m.group(1).strip()

        if "platform_or_website" not in parsed:
            m = re.search(
                r"\.\s*(GitHub|Zenodo|Figshare|OSF|Kaggle)\b", raw, re.IGNORECASE
            )
            if m:
                parsed["platform_or_website"] = m.group(1)

        if "location" not in parsed and parsed.get("title"):
            title_escaped = re.escape(parsed["title"])
            loc_match = re.search(
                title_escaped
                + r"\s*(?:\([^)]*\))?\s*,\s*([A-Z][a-zA-Z\s]{2,25}?)(?:\s*,|\s*$)",
                raw,
            )
            if loc_match:
                candidate = loc_match.group(1).strip()
                is_publisher_keyword = bool(
                    re.search(
                        r"\bpress\b|\bpublish\b|\bbooks?\b|\bgroup\b"
                        r"|\bwiley\b|\belsevier\b|\bspringer\b|\barnold\b"
                        r"|\bwilkins\b|\bmedical\b|\bscience\b|\bphysics\b",
                        candidate,
                        re.IGNORECASE,
                    )
                )
                if (
                    candidate
                    and len(candidate.split()) <= 3
                    and not re.search(r"[\d&]", candidate)
                    and not is_publisher_keyword
                    and not re.match(
                        r"^(vol|pp|ed|no|ch|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
                        candidate,
                        re.IGNORECASE,
                    )
                ):
                    parsed["location"] = candidate
                    parsed["location_source"] = "regex"

        return parsed

    def parse_all_citations(self, raw_references: List[str]) -> List[Dict]:
        """Parse tất cả raw citations thành structured data, bổ sung field thiếu từ raw text."""
        results: List[Dict] = []
        for idx, raw_text in enumerate(raw_references, 1):
            print(f"  Đang parse citation {idx}/{len(raw_references)}...")
            parsed = self.parse_citation(raw_text)
            parsed = self.fill_missing_from_raw(parsed)
            results.append(parsed)
        return results

    def print_references(self, references: List[str]) -> None:
        """In danh sách references ra console."""
        if not references:
            print("Không tìm thấy references nào trong bài báo.")
            return
        print(f"\n{'=' * 80}")
        print(f"Tìm thấy {len(references)} references:")
        print(f"{'=' * 80}\n")
        for idx, ref in enumerate(references, 1):
            print(f"[{idx}] {ref}")
            print()

    def _coerce_year_int(self, year_val: Any) -> Optional[int]:
        if not year_val:
            return None
        s = str(year_val)
        m = re.search(r"(1[89]\d{2}|20\d{2})", s)
        return int(m.group(1)) if m else None

    def _authors_for_schema(self, flat: Dict) -> Optional[list]:
        """Ưu tiên authors_structured từ TEI; fallback authors (list str); lọc fake authors."""
        _meta_tokens: set = set()
        for _v in [
            flat.get("title"),
            flat.get("location"),
            flat.get("publisher"),
            flat.get("journal"),
            flat.get("proceedings_title"),
        ]:
            if _v:
                _meta_tokens.add(_v.lower().strip())
                for _tok in re.split(r"[\s,;&]+", _v):
                    if len(_tok) > 3:
                        _meta_tokens.add(_tok.lower().strip())
        raw_text_lower = (flat.get("raw_text") or "").lower()

        def _is_fake(a: dict) -> bool:
            given = (a.get("given_name") or "").strip()
            family = (a.get("family_name") or "").strip()
            full = f"{given} {family}".strip().lower()
            if family.lower() in _meta_tokens or full in _meta_tokens:
                return True
            if not given:
                if re.search(r"[&\d]", family):
                    return True
                if len(family.split()) > 2:
                    return True
                if re.search(r"&\s+" + re.escape(family.lower()), raw_text_lower):
                    return True
            if re.search(r"&", given):
                return True
            return False

        raw_authors: Optional[list] = None
        if flat.get("authors_structured"):
            raw_authors = flat["authors_structured"]
        elif flat.get("authors") and isinstance(flat["authors"], list):
            out: List[Dict[str, str]] = []
            for name in flat["authors"]:
                name = (name or "").strip()
                if not name:
                    continue
                parts = name.split()
                if len(parts) == 1:
                    out.append({"given_name": parts[0], "family_name": ""})
                else:
                    out.append({"given_name": " ".join(parts[:-1]), "family_name": parts[-1]})
            raw_authors = out or None
        if not raw_authors:
            return None
        filtered = [a for a in raw_authors if not _is_fake(a)]
        return filtered if filtered else None

    def _classify_schema_type(self, flat: Dict) -> str:
        raw = (flat.get("raw_text") or "").lower()
        has_url = bool(flat.get("url")) or bool(re.search(r"https?://", raw))
        has_access = bool(flat.get("access_date")) or bool(re.search(r"\baccessed\b|\bretrieved\b", raw))
        has_arxiv = bool(flat.get("arxiv_id")) or ("arxiv" in raw) or ("preprint" in raw)
        has_isbn = bool(flat.get("isbn")) or ("isbn" in raw)
        is_thesis = bool(re.search(r"\bthesis\b|\bdissertation\b", raw))
        is_tech_report = bool(re.search(r"\btechnical\s+report\b|\breport\s+no\b", raw))
        has_journal = bool(flat.get("journal"))
        has_proceedings = bool(flat.get("proceedings_title")) or bool(re.search(r"\bin:\b", raw))
        if has_url and has_access:
            return "web_resource"
        if has_arxiv:
            return "preprint"
        if is_thesis:
            return "thesis"
        if is_tech_report:
            return "technical_report"
        if has_isbn:
            return "book"
        if has_journal:
            return "journal_article"
        if has_proceedings:
            return "conference_paper"
        publisher_str = (flat.get("publisher") or "").lower()
        is_book = bool(
            re.search(
                r"\bpress\b|\bpublish\b|\bedition\b|\bpublisher\b"
                r"|\bspringer\b|\bwiley\b|\belsevier\b|\bcambridge\b|\boxford\b|\barnold\b",
                publisher_str + " " + raw,
            )
        )
        if is_book:
            return "book"
        return "other"

    def _extract_conference_name(self, flat: Dict) -> Optional[str]:
        raw = flat.get("raw_text") or ""
        m = re.search(r"\(([^)]+)\)", raw)
        if m:
            cand = m.group(1).strip()
            if 2 <= len(cand) <= 25:
                return cand
        return None

    def format_output_schema(self, flat: Dict, idx: int) -> Dict:
        """Xuất đúng schema: ref_id, type, raw_text, parsed{...}"""
        ref_id = f"R{idx}"
        raw_text = flat.get("raw_text", "")
        year_int = self._coerce_year_int(flat.get("year"))
        authors = self._authors_for_schema(flat)
        schema_type = self._classify_schema_type(flat)

        if schema_type == "journal_article":
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "journal_name": flat.get("journal"),
                "volume": flat.get("volume"),
                "issue": flat.get("issue"),
                "pages_or_article_number": flat.get("pages") or flat.get("pages_or_article_number"),
                "year": year_int,
                "doi": flat.get("doi"),
            }
        elif schema_type == "conference_paper":
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "conference_name": self._extract_conference_name(flat),
                "proceedings_title": flat.get("proceedings_title"),
                "location": flat.get("location"),
                "pages": flat.get("pages"),
                "year": year_int,
                "publisher": flat.get("publisher"),
                "doi": flat.get("doi"),
            }
        elif schema_type == "preprint":
            arxiv_id = flat.get("arxiv_id")
            identifier = f"arXiv:{arxiv_id}" if arxiv_id else None
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "repository_name": "arXiv" if arxiv_id or ("arxiv" in (raw_text or "").lower()) else None,
                "identifier": identifier,
                "year": year_int,
                "version": flat.get("version"),
            }
        elif schema_type == "book":
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "edition": flat.get("edition"),
                "publisher": flat.get("publisher"),
                "location": flat.get("location"),
                "year": year_int,
                "isbn": flat.get("isbn"),
            }
        elif schema_type == "thesis":
            author_obj = authors[0] if authors else None
            parsed = {
                "author": author_obj,
                "title": flat.get("title"),
                "institution": flat.get("institution"),
                "degree_type": flat.get("degree_type"),
                "year": year_int,
            }
        elif schema_type == "technical_report":
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "institution_or_organization": flat.get("publisher") or (authors[0]["given_name"] if authors else None),
                "report_number": flat.get("report_number"),
                "year": year_int,
            }
        elif schema_type == "web_resource":
            parsed = {
                "organization": (authors[0]["given_name"] if authors else None),
                "title": flat.get("title"),
                "platform_or_website": flat.get("platform_or_website"),
                "year": year_int,
                "access_date": flat.get("access_date"),
                "url_or_doi": flat.get("url") or flat.get("doi"),
                "version": flat.get("version"),
            }
        else:
            parsed = {
                "authors": authors,
                "title": flat.get("title"),
                "year": year_int,
                "location": flat.get("location"),
                "doi": flat.get("doi"),
            }
        return {"ref_id": ref_id, "type": schema_type, "raw_text": raw_text, "parsed": parsed}
