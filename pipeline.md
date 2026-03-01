### Pipeline Architecture for Reference Verification

A high-level textual diagram.

```
[Input: PDF Paper] 
  ↓
[Extraction Stage] 
  - Preprocessing (PDF to Text/XML)
  - Parsing (GROBID + Optional Routing)
  - Output: List of Extracted References in JSON
  ↓
[Verification Stage] 
  - Normalization
  - Fetching External Metadata
  - Field-Level Similarity Computation
  - Verdict Assignment & Final Scoring
  - Output: Verified References in Extended JSON
  ↓
[Output: Verified Reference List + Benchmarks]
  ↓ (Parallel)
[Benchmarking: Extraction & Verification Metrics]
```

## Extraction Stage

This stage focuses on accurately extracting bibliographic references from the PDF's reference section. 

**Detailed Workflow:**

1. **Preprocessing**:
   
    - Convert PDF to structured text or XML using libraries like PDFBox or Poppler (fallback to Tesseract OCR for image-based PDFs).
    - Detect and isolate the reference section via heuristics (e.g., keyword search for "References" or "Bibliography", page layout analysis using bounding boxes).
    - Handle multi-column layouts by applying column detection algorithms (e.g., via OpenCV for visual segmentation).
   
    **Example Preprocessing Output**:
    ```json
    {
        "preprocessing_metadata": {
        "method": "pdfbox",
        "ocr_used": false,
        "extraction_confidence": 0.94,
        "reference_section": {
            "start_page": 12,
            "end_page": 15,
            "start_line": 245,
            "end_line": 423,
            "detection_method": "keyword+numbering",
            "detection_confidence": 0.92
        },
        "layout": {
            "columns": 2,
            "reading_order": "column-first"
        },
        "warnings": [
            {"code": "W101", "message": "Low OCR confidence on page 13 (68%)"}
        ]
        },
        "raw_reference_text": "[1] Vaswani, A., et al. (2017). Attention is All You Need...\n[2] ..."
    }
    ```

2. **Parsing with GROBID/Anystyle**:
   
    Extract structured bibliographic metadata from raw reference text using GROBID's machine learning models or Anystyle.
   
    **TEI XML Parsing**:
      
    - **XML Structure**: GROBID outputs Text Encoding Initiative (TEI) XML
        ```xml
        <biblStruct>
          <analytic>
            <title level="a">Attention is All You Need</title>
            <author><persName><forename>Ashish</forename><surname>Vaswani</surname></persName></author>
            ...
          </analytic>
          <monogr>
            <title level="m">Advances in Neural Information Processing Systems</title>
            <imprint><date type="published" when="2017">2017</date></imprint>
          </monogr>
          <idno type="arXiv">1706.03762</idno>
        </biblStruct>
        ```
      
    - **Extraction Logic** (use `xml.etree.ElementTree` or `lxml`).
    - **Confidence Scores**: GROBID provides per-field confidence in XML attributes
        - Extract: `<title coords="..." conf="0.92">Title</title>`
        - Aggregate: `extraction_confidence = mean([field_confidences])`

3. **Output Formatting**:
   
   Structure extracted references into consistent, machine-readable JSON format with comprehensive metadata.
   
    **JSON Schema Definition**:
      
      ```json
      {
        "extraction_metadata": {
          "version": "1.0.0",
          "timestamp": "2026-02-12T10:00:00Z",
          "paper_id": "input_paper_123",
          "source_pdf": "path/to/paper.pdf",
          "preprocessing": {
            "method": "pdfbox",
            "ocr_used": false,
            "confidence": 0.94
          },
          "parsing": {
            "engine": "grobid",
            "version": "0.7.3",
            "model": "citation-default.wapiti"
          },
          "statistics": {
            "total_references": 45,
            "successfully_parsed": 42,
            "partially_parsed": 2,
            "failed": 1,
            "average_confidence": 0.87
          }
        },
        "references": [
          {
            "id": "1",
            "raw_text": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).",
            
            "parsed_data": {
              "title": "Attention is all you need",
              
              "authors": ["Vaswani Ashish", "Shazeer Noam", "et al."],              
              "year": 2017,
              
              "venue": {
                "raw": "Advances in neural information processing systems",
                "type": "conference",
                "normalized": "Neural Information Processing Systems",
                "abbreviation": "NeurIPS"
              },
              
              "pages": "5998-6008",
              "volume": null,
              "issue": null,
              "publisher": null,
              
              "identifiers": {
                "doi": "10.5555/3295222.3295349",
                "arxiv_id": "1706.03762",
                "isbn": null,
                "pmid": null,
                "pmc_id": null,
                "url": "https://arxiv.org/abs/1706.03762"
              }
            },
            
            "extraction_quality": {
              "extraction_confidence": 0.92,
              "extraction_method": "grobid",
              "completeness_score": 0.95,
              "format_validity_score": 1.0,
              
              "field_confidences": {
                "title": 0.94,
                "authors": 0.91,
                "year": 0.99,
                "venue": 0.88,
                "identifiers": 0.90
              },
              
              "issues": [],
              "warnings": []
            }
          }
          // Additional references...
        ]
      }
      ```
   
    **Quality Flagging**:
      
      - **Issue Flags** (actionable problems):
        - `"missing_title"`: Title is null or <5 chars
        - `"missing_authors"`: No authors extracted
        - `"missing_year"`: Year is null
        - `"invalid_year"`: Year out of range
        - `"low_confidence"`: extraction_confidence <0.6
        - `"partial_parsing"`: Multiple fields missing
        - `"potential_duplicate"`: High similarity with previous reference
      
      - **Warning Flags** (non-critical issues):
        - `"no_identifiers"`: DOI and arXiv ID both missing
        - `"no_venue"`: Venue field empty
        - `"abbreviated_authors"`: Author names abbreviated (e.g., "A. Smith")
        - `"et_al_present"`: Author list ends with "et al."
        - `"fallback_extraction"`: GROBID failed, regex fallback used

### Extraction Benchmark

Rigorously evaluate the extraction system's accuracy, robustness, and performance across diverse reference formats and quality levels.

To evaluate extraction quality, we adopt and extend GROBID's benchmarking framework [src](https://grobid.readthedocs.io/en/latest/benchmarks/Benchmarking-biorxiv/), focusing on field-level and overall metrics with comprehensive error analysis.

#### 1. **Dataset Construction**

**Ground Truth Annotation Schema**:
```json
{
  "reference_id": "ref_001",
  "source_paper": "paper_123.pdf",
  "raw_text": "Vaswani, A., et al. (2017). Attention is all you need...",
  "ground_truth": {
    "title": "Attention is All You Need",
    "title_normalized": "attention is all you need",
    "authors": [
      {"first": "Ashish", "last": "Vaswani"},
      {"first": "Noam", "last": "Shazeer"},
      "..." 
    ],
    "year": 2017,
    "venue": {
      "name": "Neural Information Processing Systems",
      "type": "conference",
      "abbreviation": "NeurIPS"
    },
    "identifiers": {
      "doi": "10.5555/3295222.3295349",
      "arxiv_id": "1706.03762"
    },
    "pages": "5998-6008"
  },
  "metadata": {
    "quality_level": "clean",
    "citation_style": "numbered",
    "venue_type": "conference",
    "language": "english",
    "era": "modern",
    "annotator_id": "expert_02",
    "annotation_date": "2026-01-20",
    "annotation_confidence": 1.0,
    "ground_truth_source": "dblp"
  }
}
```

#### 2. **Matching Methods**

Define how extracted fields are compared with ground truth to determine correctness.

a. **Strict Matching** (Exact equality):
   - **Definition**: Byte-for-byte identical strings
   - **Application**: Year (must be exact), DOI (must be exact)
   - **Implementation**: `extracted == ground_truth`
   - **Use Case**: High-precision fields where no variation acceptable

b. **Soft Matching** (Normalized comparison):
   - **Definition**: Strings match after normalization
   - **Normalization Steps**:
     1. Lowercase: `text.lower()`
     2. Remove punctuation: Keep only alphanumeric + spaces
     3. Collapse whitespace: `re.sub(r'\s+', ' ', text).strip()`
     4. Remove diacritics: `unidecode(text)` (e.g., "ü" → "u")
   - **Application**: Title, venue name
   - **Example**: "Attention Is All You Need" matches "attention is all you need"

c. **Levenshtein Matching** (Edit distance):
   - **Definition**: Measure minimum edits (insert/delete/substitute) to transform one string into another
   - **Normalized Distance**: `1 - (levenshtein_distance / max(len(s1), len(s2)))`
   - **Threshold**: ≥ 0.80 similarity = match
   - **Application**: Title, authors (with typos/OCR errors)
   - **Library**: `python-Levenshtein` (fast C implementation)
   - **Example**: "Atention is All You Need" (typo) → 0.96 similarity → match

d. **Ratcliff/Obershelp Matching** (Sequence-based):
   - **Definition**: Find longest common subsequence, recursively match around it
   - **Computation**: `difflib.SequenceMatcher(None, s1, s2).ratio()`
   - **Threshold**: ≥ 0.95 similarity = match
   - **Application**: Robust to word reordering
   - **Example**: "All You Need is Attention" vs "Attention is All You Need" → high score

e. **Author Matching** (Special handling for lists):
   - **Last Name Only**: Extract last names, compute Jaccard similarity
     - `similarity = len(set(E_last) ∩ set(GT_last)) / len(set(E_last) ∪ set(GT_last))`
     - Threshold: ≥ 0.75 = match
   - **Full Name**: Match if first initial + last name match for ≥80% of authors
   - **Ordering Flexibility**: Don't penalize for different author order
   - **Et al. Handling**: If ground truth has 8 authors but extracted has 6 + "et al.", match if first 6 match

f. **Identifier Matching** (Special rules):
   - **DOI**: Exact match after normalization (lowercase, remove "https://doi.org/" prefix)
   - **arXiv**: Match base ID, ignore version (e.g., "1706.03762v1" matches "1706.03762")
   - **ISBN**: Convert ISBN-10 to ISBN-13, then exact match

#### 3. **Evaluation Metrics**

**Field-Level Metrics** (Per-field evaluation):

a. **Binary Classification Metrics** (for each field):
   
   - **Precision**: `P = TP / (TP + FP)`
     - Interpretation: Of all extracted fields, what fraction are correct?
     - High precision: Low false alarm rate
   
   - **Recall**: `R = TP / (TP + FN)`
     - Interpretation: Of all ground truth fields, what fraction were extracted?
     - High recall: Low miss rate
   
   - **F1-Score**: `F1 = 2 * P * R / (P + R)`
     - Harmonic mean balances precision and recall
     - Primary metric for overall performance
   
   - **Accuracy**: `Acc = (TP + TN) / (TP + FP + FN + TN)`
     - Overall correctness rate
     - Less meaningful for imbalanced data

b. **Example for Per-Field Reporting** (per matching method):
   
   | Field        | Precision | Recall | F1    | Accuracy | Matches | Mismatches | Missing |
   |--------------|-----------|--------|-------|----------|---------|------------|---------|
   | **Title**    | 0.94      | 0.96   | 0.95  | 0.93     | 2850    | 120        | 30      |
   | **Authors**  | 0.88      | 0.91   | 0.89  | 0.86     | 2730    | 180        | 90      |
   | **Year**     | 0.99      | 0.99   | 0.99  | 0.99     | 2970    | 15         | 15      |
   | **Venue**    | 0.86      | 0.89   | 0.87  | 0.84     | 2670    | 210        | 120     |
   | **DOI**      | 0.92      | 0.85   | 0.88  | 0.91     | 1700    | 80         | 220     |
   | **arXiv ID** | 0.95      | 0.88   | 0.91  | 0.94     | 440     | 20         | 40      |

c. **Example for Matching Method Comparison (F1)**:
   
   | Field   | Strict | Soft  | Levenshtein (>=0.8) | Ratcliff (>=0.95) |
   |---------|--------|-------|--------------------|------------------|
   | Title   | 0.89   | 0.95  | 0.96               | 0.94             |
   | Venue   | 0.78   | 0.87  | 0.88               | 0.86             |
   
   - **Analysis**: Soft matching +6% F1 over strict for titles
   - **Recommendation**: Use soft matching for text fields, strict for identifiers

d. **Weighted Overall Metrics**:
   
   - **Macro-averaged F1**: `Mean(F1_title, F1_authors, F1_year, F1_venue, F1_identifiers)`
     - Treats all fields equally
     - Good for field-level understanding
   
   - **Micro-averaged F1**: `F1(Σ TP, Σ FP, Σ FN)` across all fields
     - Treats all instances equally
     - Dominated by frequent fields
   
   - **Weighted F1**: `Σ (weight_field * F1_field)`
     - Weights by field importance:
       - Title: 0.35, Authors: 0.30, Year: 0.15, Venue: 0.10, Identifiers: 0.10
     - Aligns with downstream verification importance

## Verification Stage

This stage verifies extracted references against external authoritative sources (e.g., CrossRef, Semantic Scholar, arXiv API) to detect inaccuracies, fabrications, or incompleteness. It involves normalization to handle variations, fetching canonical metadata, multi-metric similarity computation, and verdict assignment.

**Detailed Workflow:**
1. **Normalization**:
   
   Standardize extracted reference fields to canonical forms for consistent comparison with external metadata.
   
   **Pre-processing Validation**:
   - Check for null/empty fields and log warnings
   - Detect encoding issues (UTF-8, Latin-1) and normalize to UTF-8
   - Remove zero-width characters and control characters (regex: `[\x00-\x1F\x7F-\x9F]`)
   - Trim excessive whitespace (replace `\s+` with single space)
   
   **Field-Specific Normalization** (authors, title, year, venue, identifiers):
    - Authors: Split into first/last names, remove affiliations, normalize abbreviations (e.g., "A. Vaswani" → "Ashish Vaswani" using name entity resolution).
    - Title: Lowercase, remove punctuation, stem words (e.g., via NLTK Porter Stemmer).
    - Year: Convert to integer, handle ranges (e.g., "2017-2018" → 2017).
    - Venue: Map abbreviations to full names (e.g., "NIPS" → "NeurIPS" using a lookup table or Wikidata API).
    - Identifiers: Validate formats (e.g., DOI checksum).

   **Optimizations**:
   - Parallelize normalization across references using multiprocessing (Python `joblib` or `concurrent.futures`)
   - Cache normalized venue mappings and author name expansions in Redis/Memcached
   - Use batch processing for embeddings (process 100+ titles at once)

2. **Fetching External Metadata**:
   
   Retrieve authoritative reference metadata from external sources to validate extracted data.
   
   **API Source Prioritization** (Sequential fallback strategy):
   
    a. **Primary: Identifier-Based Retrieval** (highest accuracy)
      - **DOI via CrossRef**
      - **arXiv ID via arXiv API**
      - **ISBN via OpenLibrary/Google Books**
      - **PMID via PubMed E-utilities**
   
    b. **Secondary: Fuzzy Search** (if no valid identifier or identifier lookup fails, combine title + authors + year for fuzzy search):
      - **Semantic Scholar**
      - **OpenAlex**
      - **Google Scholar (Cautious Use)**
   
    c. **Tertiary: Fallback APIs**:
      - **DBLP**: For CS papers, query `https://dblp.org/search/publ/api?q={title}&format=json`
      - **Europe PMC**: For life sciences, `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={title}`
      - **BASE (Bielefeld Academic Search Engine)**: For open access papers
   
   **Query Construction Best Practices**:
    - **Title-based**: Wrap in quotes for exact phrase matching, remove special chars
    - **Author-based**: Use last names only (first names may have abbreviations/variations)
    - **Combined**: Use boolean operators: `title AND author AND year` (if API supports)
    - **Incremental relaxation**: If no results, progressively remove constraints:
        1. Full query: title + authors + year
        2. Title + year
        3. Title only (if unique enough, e.g., >5 words)
   
   **Response Handling**:
   
   a. **No Results**:
      - Log as `fetch_status="not_found"`
      - Attempt alternative spellings (e.g., British vs. American: "behaviour" → "behavior")
      - Check if title contains special characters that broke query (re-query with cleaned title)
      - Final verdict: `"unfetchable"` with reason `"no_match_found"`
   
   b. **Single Result**:
      - Accept if similarity score >0.7 (preliminary title Levenshtein)
      - Validate: Check year match (±2 years acceptable for preprints/revisions)
      - Log as `fetch_status="single_match", fetch_confidence=high`
   
   c. **Multiple Results** (Disambiguation required):
      - **Ranking Strategy** (score each candidate):
        1. Title similarity: Levenshtein distance on normalized titles (weight: 0.5)
        2. Author overlap: Jaccard similarity on last names (weight: 0.3)
        3. Year match: 1.0 if exact, 0.5 if ±1 year, 0 otherwise (weight: 0.2)
        4. Venue similarity: If extracted venue available, fuzzy match (weight: 0.1 if available)
        5. Citation count: Tiebreaker for highly-cited vs. obscure papers (prefer higher citations)
      - **Selection**:
        - Top-1 if score >0.8: Accept as match
        - Top-1 score 0.6-0.8: Flag as `fetch_status="ambiguous_match"`
        - Top-1 score <0.6: No match, log as `fetch_status="no_confident_match"`
      - **Logging**: Store top-3 candidates with scores for manual review/debugging
   
   **Caching Strategy** (Critical for performance): use local cache (SQLite/PostgreSQL) and in-memory caches to minimize redundant API calls.

   **Output Structure** (per reference):
   ```json
   {
     "fetched_data": {
       "source_api": "crossref",
       "fetch_status": "single_match",
       "fetch_confidence": 0.95,
       "fetch_timestamp": "2026-02-12T10:30:00Z",
       "title": "Attention is All You Need",
       "authors": [{"first": "Ashish", "last": "Vaswani"}, ...],
       "year": 2017,
       "venue": {"name": "Neural Information Processing Systems", "type": "conference"},
       "identifiers": {"doi": "10.5555/3295222.3295349", "arxiv_id": "1706.03762"},
       "citation_count": 70000,
       "alternative_matches": [...]  // Top-3 if ambiguous
     }
   }
   ```

3. **Field-Level Similarity Computation**:
   
   Quantify similarity between extracted (E) and fetched (F) data for each field to identify discrepancies.
   
   **General Principles**:
   - Compute multiple similarity metrics per field for robustness
   - Normalize all scores to [0, 1] range
   - Use field-specific thresholds based on expected variation
   - Store individual metric scores for debugging and threshold tuning
   
   **Field-Specific Algorithms**:
   
   a. **Title Similarity** (Ensemble of metrics):
      
      - **String-Based Metrics**:
        1. **Levenshtein Distance** (Edit distance):
           - Normalized: `1 - (edit_distance / max(len(E_title), len(F_title)))`
           - Use `python-Levenshtein` for fast C implementation
           - Weight: 0.2
        2. **Jaro-Winkler** (Prefix-sensitive):
           - Good for detecting transposed characters
           - Library: `jellyfish.jaro_winkler_similarity`
           - Weight: 0.15
        3. **Ratcliff/Obershelp** (Gestalt pattern matching):
           - Library: `difflib.SequenceMatcher(None, E_title, F_title).ratio()`
           - Weight: 0.15
      
      - **Token-Based Metrics**:
        1. **Jaccard Similarity** (Set overlap):
           - Tokenize: Split on whitespace, remove stop words
           - `len(E_tokens ∩ F_tokens) / len(E_tokens ∪ F_tokens)`
           - Weight: 0.15
        2. **TF-IDF Cosine Similarity**:
           - Vectorize titles using scikit-learn `TfidfVectorizer`
           - Compute: `cosine_similarity(E_vector, F_vector)`
           - Weight: 0.15
      
      - **Semantic Metrics** (Embedding-based):
        1. **SciBERT/Sentence-BERT Cosine**:
           - Use pre-computed embeddings from normalization stage
           - Compute: `cosine(E_embedding, F_embedding)` using `scipy.spatial.distance.cosine`
           - Best for capturing semantic similarity ("neural network" ≈ "deep learning model")
           - Weight: 0.2
      
      - **Ensemble Score**:
        - `title_score = 0.2*lev + 0.15*jaro + 0.15*ratcliff + 0.15*jaccard + 0.15*tfidf + 0.2*semantic`
        - Thresholds: >0.95 = exact match, 0.8-0.95 = fuzzy match, <0.8 = mismatch
      
      - **Edge Cases**:
        - Missing title (either E or F): score = 0, verdict = "incomplete"
        - Title length ratio >3:1 (e.g., "DL" vs. "Deep Learning Fundamentals"): suspect abbreviation, apply fuzzy threshold
        - Subtitle mismatch: If main title matches but subtitle differs, score = 0.9
   
   b. **Author Similarity** (List comparison with ordering flexibility):
      
      - **Exact Match**: Check if lists are identical (full names, same order)
        - Score: 1.0 if exact, else proceed to fuzzy
      
      - **Last Name Jaccard**:
        - Extract last names from both lists: `E_last_names = {e.last for e in E_authors}`
        - Compute: `len(E_last ∩ F_last) / len(E_last ∪ F_last)`
        - Weight: 0.4
      
      - **First Name Initial Match** (among matched last names):
        - For each matched last name, check if first initial matches
        - Score: (matched_initials / matched_last_names)
        - Weight: 0.2
      
      - **Author Sequence Similarity** (order matters):
        - Compare first 3 authors only (many papers list only first few)
        - Use sequence alignment (Needleman-Wunsch) or simple position-based scoring
        - `sum(author_match(E[i], F[i]) for i in range(min(3, len(E), len(F))))` / 3
        - Weight: 0.2
      
      - **"Et al." Handling**:
        - If E has "et al." (truncated), compare only available authors
        - Compute partial score: E authors must be subset of F authors
        - Penalty: Reduce score by 0.1 if truncation detected
      
      - **Ensemble Score**:
        - `author_score = 0.4*jaccard + 0.2*initial_match + 0.2*sequence + 0.2*full_name_match`
        - Adjust if truncated: `author_score *= 0.9`
      
      - **Thresholds**: >0.9 = match, 0.6-0.9 = soft_mismatch (some authors missing/misspelled), <0.6 = hard_mismatch
      
      - **Edge Cases**:
        - Single author mismatch: score = 1.0 if exact, 0.5 if first initial + last name match, 0 otherwise
        - Corporate/group authors (e.g., "WHO"): Use exact string match or fuzzy match on organization name
        - Name order variants (e.g., "First Last" vs. "Last, First"): Normalize during comparison
        - Non-Latin names: Use transliterated version from normalization
   
   c. **Year Similarity** (Binary with tolerance):
      
      - **Exact Match**: `year_score = 1.0 if E_year == F_year else ...`
      
      - **Tolerance Window** (for preprints/revisions):
        - Acceptable difference: ±1 year → score = 0.9
        - Acceptable difference: ±2 years → score = 0.7 (flag for review)
        - Difference >2 years → score = 0 (hard_mismatch)
      
      - **Special Cases**:
        - E_year is None: score = 0, verdict = "incomplete"
        - F_year has multiple (e.g., preprint vs. published): Use published date, score = 1.0 if matches either
        - Forthcoming/in press: If F_year = current_year+1 and E_year = current_year, score = 0.9
      
      - **Threshold**: Binary metric, no fuzzy matching needed
   
   d. **Venue Similarity** (Hierarchical matching):
      
      - **Exact String Match** (normalized):
        - Compare normalized venue names (from normalization stage)
        - Score: 1.0 if exact match
      
      - **Abbreviation Match**:
        - Check if E_venue (abbr) expands to F_venue (full) or vice versa
        - Use lookup table: `{"NIPS": "NeurIPS", "ICML": "International Conference on Machine Learning"}`
        - Score: 0.95 if found in table
      
      - **Fuzzy String Match**:
        - Apply Levenshtein/Jaro-Winkler on venue strings
        - Threshold: >0.85 → score = 0.9 (likely same venue with minor differences)
      
      - **Semantic Match** (embedding-based):
        - Embed venue names using Sentence-BERT
        - Compute cosine similarity
        - Good for detecting "Proceedings of ACL" vs. "ACL Conference"
        - Threshold: >0.9 → score = 0.85
      
      - **Venue Type Match**:
        - If both have venue_type, check consistency (conference vs. journal)
        - Mismatch penalty: Reduce score by 0.2
      
      - **Ensemble Score**:
        - Use highest score from [exact, abbreviation, fuzzy, semantic]
        - Apply type penalty if applicable
      
      - **Thresholds**: >0.95 = match, 0.8-0.95 = soft_mismatch (abbreviation/minor diff), <0.8 = hard_mismatch
      
      - **Edge Cases**:
        - Missing venue: score = 0, verdict = "incomplete"
        - Generic venues (e.g., "arXiv"): Use exact match only (score = 1.0 or 0)
        - Venue name changes over time (e.g., "NIPS" → "NeurIPS"): Maintain historical mapping
   
   e. **Identifier Similarity** (Exact with format normalization):
      
      - **DOI**:
        - Normalize: Remove prefix, lowercase
        - Exact match: score = 1.0
        - Partial match (different versions): score = 0.8 if base DOI matches (e.g., 10.1234/abc vs. 10.1234/abc.v2)
        - Mismatch: score = 0 (critical error)
      
      - **arXiv ID**:
        - Normalize: Handle old/new format, remove version
        - Exact match (ignoring version): score = 1.0
        - Version mismatch: score = 0.95 (same paper, different revision)
      
      - **ISBN**:
        - Normalize: Convert ISBN-10 to ISBN-13
        - Exact match: score = 1.0
        - Different editions: score = 0.7 if same title/authors confirmed
      
      - **Ensemble Score** (if multiple identifiers):
        - Average: `(doi_score + arxiv_score + isbn_score) / num_available_identifiers`
        - If any identifier is present in both and matches → boost overall confidence
      
      - **Thresholds**: Binary (1.0 = match, 0.95 = version diff, <0.95 = mismatch)
      
      - **Edge Cases**:
        - E has identifier but F doesn't (or vice versa): score = 0.5, verdict = "incomplete"
        - Both missing: score = 0, verdict = "incomplete" (rely on other fields)
   
   **Aggregate Scoring** (Field weights + overall score):
   
   - **Field Weights** (customizable based on use case):
     - Default: `{"title": 0.35, "authors": 0.30, "year": 0.15, "venue": 0.10, "identifiers": 0.10}`
     - High-stakes (detecting fabrication): Increase identifiers weight to 0.20, reduce venue to 0.05
     - Fuzzy matching mode (OCR errors): Increase title weight to 0.40, reduce year to 0.10
   
   - **Overall Similarity Score**:
     ```python
     overall_score = (weights["title"] * title_score + 
                      weights["authors"] * author_score + 
                      weights["year"] * year_score + 
                      weights["venue"] * venue_score + 
                      weights["identifiers"] * identifier_score)
     ```
   
   **Optimization**:
   - Precompute embeddings (SciBERT) during normalization to avoid redundant computation
   - Use vectorized operations (NumPy) for batch similarity computation
   - Cache similarity scores for reference pairs (if processing same paper multiple times)
   - Parallelize across references using multiprocessing
   
   **Error Handling**:
   - Missing fields: Assign score = 0, exclude from weighted average (renormalize weights)
   - Null/empty strings: Treat as missing
   - Exception during computation: Log error, assign score = 0, continue processing

4. **Verdict Assignment and Final Scoring**:
   
   Translate continuous similarity scores into discrete, actionable verdicts with confidence assessments.
   
   **Field-Level Verdict Assignment** (Rule-based with field-specific thresholds):
   
   a. **Verdict Categories** (5-level classification):
      - `"match"`: High confidence that fields match
      - `"soft_mismatch"`: Minor discrepancies (abbreviations, formatting, truncation)
      - `"hard_mismatch"`: Critical discrepancies indicating likely error or fabrication
      - `"incomplete"`: Field missing in extracted OR fetched data
      - `"ambiguous"`: Cannot determine with confidence (multi-match scenarios, low-quality data)
   
   b. **Field-Specific Thresholds and Rules**:
      
      **Title**:
      - `match`: score ≥ 0.95 (near-exact match)
      - `soft_mismatch`: 0.80 ≤ score < 0.95 (minor differences: punctuation, subtitle variations)
      - `hard_mismatch`: score < 0.80 (substantially different titles)
      - `incomplete`: Either E_title or F_title is null/empty
      - `ambiguous`: fetch_status = "ambiguous_match" AND score ∈ [0.70, 0.85]
      - **Special flags**: 
        - `subtitle_only_mismatch`: Main title matches but subtitle differs
        - `abbreviation_suspected`: Title length ratio >3:1
      
      **Authors**:
      - `match`: score ≥ 0.90 (all or nearly all authors match)
      - `soft_mismatch`: 0.60 ≤ score < 0.90 (some authors missing, misspelled names, et al. truncation)
      - `hard_mismatch`: score < 0.60 (different author sets)
      - `incomplete`: Either E_authors or F_authors is empty
      - `ambiguous`: Author name resolution failed for >50% of authors
      - **Special flags**:
        - `et_al_truncation`: E has "et al." and missing authors
        - `name_order_variant`: Authors present but in different order
        - `single_author_mismatch`: Only 1 author and names differ
      
      **Year**:
      - `match`: score = 1.0 (exact match)
      - `soft_mismatch`: score ∈ [0.7, 0.9] (±1-2 years, acceptable for preprints)
      - `hard_mismatch`: score = 0 (>2 years difference)
      - `incomplete`: E_year or F_year is null
      - `ambiguous`: F_year has multiple values (preprint vs. published) and unclear which to use
      - **Special flags**:
        - `preprint_adjustment`: Different year but identified as preprint vs. published version
        - `forthcoming`: Marked as "in press" or future year
      
      **Venue**:
      - `match`: score ≥ 0.95 (exact or known abbreviation)
      - `soft_mismatch`: 0.80 ≤ score < 0.95 (fuzzy match, minor variations)
      - `hard_mismatch`: score < 0.80 (different venue)
      - `incomplete`: E_venue or F_venue is null
      - `ambiguous`: Venue string too generic (e.g., "Conference") or multiple candidates
      - **Special flags**:
        - `abbreviation_match`: Matched via lookup table
        - `venue_type_mismatch`: Same name but different type (journal vs. conference)
      
      **Identifiers**:
      - `match`: score = 1.0 (exact match on DOI/arXiv/ISBN)
      - `soft_mismatch`: score ∈ [0.90, 0.99] (version differences only)
      - `hard_mismatch`: score = 0 (identifiers present but different)
      - `incomplete`: Both E and F missing identifiers, OR one present one absent (score = 0.5)
      - `ambiguous`: Identifier format invalid or checksum failed
      - **Special flags**:
        - `version_difference`: Same base ID, different version
        - `identifier_extraction_failed`: Attempted extraction from raw text but failed
   
   c. **Confidence Scoring** (Per-field confidence in verdict):
      - Factors affecting confidence:
        1. **Score proximity to threshold**: Closer to boundary → lower confidence
           - Compute: `confidence = 1 - exp(-abs(score - threshold) * 5)` (sigmoid-like)
        2. **Fetch confidence**: Propagate from fetching stage
           - If fetch_confidence < 0.8, reduce verdict confidence by 20%
        3. **Normalization quality**: If normalization had issues, reduce confidence
        4. **Metric agreement**: If multiple metrics disagree (high variance), reduce confidence
           - Compute metric score variance; if std > 0.15, flag low_confidence
      - Final field confidence: `field_confidence = base_confidence * fetch_confidence * normalization_quality`
      - Threshold: confidence < 0.70 → add warning flag
   
   d. **Verdict Output Structure** (per field):
      ```json
      {
        "title": {
          "score": 0.92,
          "verdict": "match",
          "confidence": 0.88,
          "threshold_used": 0.95,
          "metrics": {"levenshtein": 0.91, "semantic": 0.93, ...},
          "flags": [],
          "explanation": "Titles are semantically similar with minor punctuation differences."
        }
      }
      ```
   
   **Overall Verdict and Scoring**:
   
   a. **Overall Similarity Score Computation**:
      - Base formula (from section 3):
        ```python
        overall_score = sum(weights[field] * scores[field] for field in fields)
        ```
      - Adjust for missing fields (renormalize weights):
        ```python
        available_fields = [f for f in fields if verdicts[f] != "incomplete"]
        total_weight = sum(weights[f] for f in available_fields)
        overall_score = sum(weights[f] * scores[f] for f in available_fields) / total_weight
        ```
      - Apply confidence adjustment:
        ```python
        avg_confidence = mean([verdicts[f]["confidence"] for f in available_fields])
        if avg_confidence < 0.8:
            overall_score *= (0.9 + avg_confidence * 0.1)  # Penalize low confidence
        ```
      - Identifier boost (high trust in exact ID matches):
        ```python
        if identifiers_verdict == "match" and identifiers_score == 1.0:
            overall_score = min(1.0, overall_score * 1.05)
        ```
   
   b. **Overall Verdict Categories** (6-level hierarchical):
      
      - **`"verified"`**: score > 0.90 AND no hard_mismatch in any field AND avg_confidence > 0.80
        - Trusted reference, safe to use
        - Sub-category: `"strongly_verified"` if identifiers match + score > 0.95
      
      - **`"likely_verified"`**: 0.85 ≤ score ≤ 0.90 AND no hard_mismatch AND avg_confidence > 0.70
        - High likelihood correct but minor issues (soft mismatches acceptable)
      
      - **`"partially_verified"`**: 0.70 ≤ score < 0.85 OR 1 soft_mismatch + score > 0.80
        - Some fields match but significant discrepancies present
        - Requires human review for critical applications
      
      - **`"questionable"`**: 0.50 ≤ score < 0.70 OR 1 hard_mismatch OR avg_confidence < 0.60
        - Likely errors in extraction or fabrication concerns
        - Do not use without verification
      
      - **`"unverified"`**: score < 0.50 OR ≥2 hard_mismatches OR fetch_status = "no_confident_match"
        - Cannot verify reference, high error likelihood
        - Flag for manual investigation
      
      - **`"unfetchable"`**: fetch_status ∈ ["not_found", "error", "timeout"]
        - Cannot retrieve external metadata
        - Recount: May be valid but unpublished/obscure, OR may be fabricated
   
   c. **Verdict Priority Rules** (Override overall score in specific cases):
      1. **Identifier match override**: If DOI or arXiv exact match -> minimum verdict = "likely_verified" (even if other fields mismatch slightly)
      2. **Hard mismatch veto**: Any hard_mismatch in title OR authors -> maximum verdict = "questionable"
      3. **Year critical**: Year hard_mismatch + no identifier -> maximum verdict = "unverified" (likely fabrication)
      4. **Fetch failure**: If fetch_status = "error" -> verdict = "unfetchable" regardless of score
   
   d. **Explainability** (Human-readable verdict explanation):
      - Generate natural language explanation:
        ```python
        if overall_verdict == "verified":
            explanation = f"Reference verified with {overall_score:.1%} confidence. "
            if identifiers_verdict == "match":
                explanation += "Identifier match confirms authenticity. "
            if any_soft_mismatch:
                explanation += f"Minor discrepancies in {mismatch_fields} acceptable."
        elif overall_verdict == "unverified":
            explanation = f"Cannot verify reference (score: {overall_score:.1%}). "
            explanation += f"Critical mismatches in: {hard_mismatch_fields}. "
            explanation += "Manual review required."
        ```
      - Store in output JSON for debugging and user feedback
   
   **Error Handling and Edge Cases**:
   - All fields incomplete → verdict = "unfetchable", overall_score = 0
   - Extraction confidence < 0.50 → add warning: `"low_extraction_quality"`
   - Circular conflicts (e.g., A matches B, B matches C, A != C) → flag as `"ambiguous"`
   - Missing ground truth fields (e.g., F_venue = null for arXiv preprints) → don't penalize, mark as N/A


### Verification Benchmark

Rigorously evaluate verification system accuracy, reliability, and robustness using standardized protocols and diverse test datasets.

#### 1. **Dataset Construction**

**Ground Truth Annotation Schema**:
```json
{
  "reference_id": "ref_001",
  "ground_truth": {
    "title": "Attention is All You Need",
    "authors": [{"first": "Ashish", "last": "Vaswani"}, ...],
    "year": 2017,
    "venue": {"name": "NeurIPS", "type": "conference"},
    "identifiers": {"arxiv_id": "1706.03762", "doi": "..."},
    "field_verdicts": {
      "title": "match",  // Expected verdict
      "authors": "soft_mismatch",  // If extraction has issues
      ...
    },
    "overall_verdict": "verified",
    "notes": "Author list truncated in extraction"
  }
}
```

#### 2. **Evaluation Metrics**

**Field-Level Metrics** (Per-field classification):

a. **Multi-Class Classification** (for title, authors, venue):
   - Classes: `[match, soft_mismatch, hard_mismatch, incomplete, ambiguous]`
   - **Precision per class**: TP / (TP + FP)
     - Precision_match: How many predicted "match" are truly matches?
     - Critical: High precision for "hard_mismatch" (avoid false alarms)
   - **Recall per class**: TP / (TP + FN)
     - Recall_match: What fraction of true matches are detected?
   - **F1-Score per class**: Harmonic mean of precision and recall
   - **Macro-averaged F1**: Average F1 across all classes (treats classes equally)
   - **Weighted F1**: Weight by class frequency (account for imbalance)
   
b. **Binary Classification** (for year, identifiers):
   - Classes: `[match, mismatch]` (no soft category)
   - **Accuracy**: (TP + TN) / Total
   - **Precision/Recall/F1**: Standard binary metrics
   - **Year**: Target >99% accuracy (should be exact)
   - **Identifiers**: Target >95% F1 (validation may fail for obscure sources)

**Overall Verdict Metrics**:

a. **Classification Metrics** (6-class problem):
   - Classes: `[verified, likely_verified, partially_verified, questionable, unverified, unfetchable]`
   - **Accuracy**: Overall correct prediction rate
   - **Macro F1**: Average F1 across all verdict classes
   - **Weighted F1**: Account for class imbalance (most refs should be "verified")
   - **Cohen's Kappa**: Inter-rater agreement (system vs. ground truth)
     - κ > 0.80: Excellent agreement
     - κ 0.60-0.80: Good agreement

b. **Ranking Metrics** (treating overall_score as ranking):
   - **Mean Average Precision (MAP)**: Average precision across all references
   - **NDCG@k**: Normalized Discounted Cumulative Gain (top-k ranking quality)
   - **AUC-ROC**: Area under ROC curve for binary "verified" vs. "not verified"
   - **Precision@k**: Precision of top-k verified references

c. **Threshold Sensitivity Analysis**:
   - Vary overall_verdict thresholds (e.g., 0.85-0.95 for "verified")
   - Plot precision-recall curves for each threshold
   - Select optimal threshold via F1 maximization or cost-sensitive learning
   - Report: Operating point (threshold) and corresponding P/R/F1

#### 3. **Reporting Standards**

**Required Reporting Elements**:

1. **Dataset Statistics**:
   - Total references, split sizes, composition breakdown
   - Ground truth annotation protocol, inter-annotator agreement

2. **Metric Table** (Main results):
   ```
   | Field       | Precision | Recall | F1    | Accuracy |
   |-------------|-----------|--------|-------|---------|
   | Title       | 0.94      | 0.96   | 0.95  | 0.93    |
   | Authors     | 0.88      | 0.91   | 0.89  | 0.86    |
   | Year        | 0.99      | 0.99   | 0.99  | 0.99    |
   | Venue       | 0.86      | 0.89   | 0.87  | 0.84    |
   | Identifiers | 0.92      | 0.85   | 0.88  | 0.91    |
   | **Overall** | **0.91**  | **0.92** | **0.91** | **0.89** |
   ```
