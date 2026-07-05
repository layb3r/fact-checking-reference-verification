from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Identifiers:
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]


@dataclass
class ExtractedCitation:
    ref_id: str
    raw_text: Optional[str]
    title: Optional[str]
    authors: List[str]
    venue: Optional[str]
    year: Optional[int]
    identifiers: Identifiers