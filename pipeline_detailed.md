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
   
   **Purpose**: Convert PDF to machine-readable format and isolate the reference section for accurate parsing.
   
   a. **PDF to Text/XML Conversion**:
      
      - **Primary Method - Native Text Extraction**:
        - **PDFBox** (Java-based, Apache license):
          - Command: `pdfbox export:text -i input.pdf -o output.txt -encoding UTF-8`
          - Advantages: Preserves text structure, handles complex layouts
          - Limitations: Fails on image-based PDFs, struggles with non-standard fonts
        - **Poppler/pdftotext** (C++ library with Python bindings):
          - Command: `pdftotext -layout -enc UTF-8 input.pdf output.txt`
          - Advantages: Fast (~0.5s per page), maintains spatial layout via `-layout` flag
          - Use `-bbox` flag for bounding box coordinates (useful for column detection)
        - **PyMuPDF (fitz)** (Python):
          - Fast extraction with block-level metadata
          - Code: `doc = fitz.open('input.pdf'); text = page.get_text('text')`
          - Can extract images, fonts, and structural information
      
      - **Fallback - OCR for Image-Based/Scanned PDFs**:
        - **Detection Strategy**: If extracted text is <100 chars per page or has >30% non-ASCII, trigger OCR
        - **Tesseract OCR** (v5+, with LSTM neural network):
          - Prerequisites: Convert PDF to images (300 DPI minimum)
          - Command: `tesseract input.png output -l eng --psm 1 --oem 1`
          - `--psm 1`: Automatic page segmentation with OSD (Orientation and Script Detection)
          - `--oem 1`: LSTM neural net mode (highest accuracy)
          - Post-processing: Use confidence scores to flag low-quality extractions (<70% confidence)
        - **Cloud OCR APIs** (for critical applications):
          - Google Cloud Vision API: Higher accuracy on academic papers
          - Azure Computer Vision: Good for multi-column layouts
          - Fallback order: Tesseract → Google Cloud → Azure
      
      - **Quality Assessment**:
        - Compute extraction confidence: `confidence = 1 - (special_chars_rate * 0.5 + missing_words_rate * 0.5)`
        - Flag pages with confidence <0.7 for manual review
        - Log character error statistics: substitutions, deletions, insertions
   
   b. **Reference Section Detection and Isolation**:
      
      - **Heuristic-Based Detection** (multi-strategy approach):
        
        1. **Keyword Search**:
           - Search for section headers: `["References", "Bibliography", "Works Cited", "Literature Cited", "References and Notes"]`
           - Regex: `(?i)^\s*(References|Bibliography|Works Cited|Literature)\s*$` (case-insensitive, line start)
           - Language-specific: Add non-English variants (e.g., "Références" for French, "Literatur" for German)
           - Confidence: High if exact match, medium if fuzzy match
        
        2. **Numbering Pattern Recognition**:
           - Detect reference numbering: `[1]`, `1.`, `(1)`, `[Vaswani17]`
           - Regex: `^\s*[\[\(]?\d+[\]\)\.]\s+` (start of line, optional brackets/parens)
           - Count consecutive numbered lines: If >5 consecutive, likely reference section
           - Handle restart: Some papers restart numbering in appendix
        
        3. **Citation Format Detection**:
           - Author-year format: `Vaswani et al. (2017)` or `(Vaswani et al., 2017)`
           - Regex: `([A-Z][a-z]+)(,?\s+[A-Z]\.)?\s+(et al\.)?\s+\(?\d{4}\)?`
           - Count matches per paragraph: If >80% paragraphs match, likely references
        
        4. **Page Layout Analysis**:
           - References often start on new page, have different formatting
           - Check for: Smaller font size, single spacing, hanging indents
           - Use PDF structure: Check for TOC entries pointing to "References"
        
        5. **Structural Heuristics**:
           - References typically in last 10-20% of paper
           - Often preceded by "Acknowledgments" or "Conclusion"
           - Rarely followed by substantial content (maybe appendix)
        
        6. **Machine Learning-Based Detection** (optional, for robustness):
           - Train binary classifier (e.g., LogisticRegression, RandomForest) on features:
             - Text density, line length variance, numbering presence, keyword match
             - Font size, spacing, indentation (from PDF metadata)
           - Training data: 500+ papers with manually annotated reference sections
           - Fallback to heuristics if classifier confidence <0.8
      
      - **Boundary Determination**:
        - **Start**: First line matching reference pattern after header detection
        - **End**: Last consecutive reference entry, or start of appendix, or EOF
        - Handle multi-section papers: Separate main references from appendix references
        - Validate: Reference section should be ≥0.5 page and ≤20% of total paper
      
      - **Error Handling**:
        - No section detected: Extract last 3 pages and attempt parsing (conservative fallback)
        - Multiple candidates: Select section with highest reference density (entries per page)
        - Empty section: Flag entire document for manual review
   
   c. **Multi-Column Layout Handling**:
      
      - **Column Detection**:
        
        1. **Text-Based Method**:
           - Analyze horizontal spacing: Detect gap >2x average inter-word space
           - Use pdftotext's `-bbox` output: Group text blocks by x-coordinate ranges
           - Threshold: Column boundary if vertical gap span >50% of page height
        
        2. **Visual/Image-Based Method** (for complex layouts):
           - **OpenCV Pipeline**:
             1. Convert PDF page to image (300 DPI)
             2. Apply Gaussian blur: `cv2.GaussianBlur(img, (5,5), 0)`
             3. Threshold to binary: `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)`
             4. Morphological closing: `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)` to merge text blocks
             5. Find contours: `cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
             6. Cluster contours by x-coordinate: Use DBSCAN or k-means (k=2 for 2-column)
           - **Page Segmentation** (via tesseract):
             - Use `--psm 1` (auto with OSD) or `--psm 3` (auto without OSD)
             - Extract layout information: `tesseract input.png output --psm 1 hocr` (HTML OCR format)
             - Parse HOCR to get column boundaries
        
        3. **PDF Structure-Based** (if available):
           - Some PDFs have embedded column structure in metadata
           - Use PyMuPDF: `page.get_text('dict')` returns blocks with bbox coordinates
           - Group blocks by x-position clustering
      
      - **Reading Order Reconstruction**:
        - **Column-first strategy** (most common): Read full left column, then right
        - **Row-first strategy** (rare): Alternate between columns row by row
        - Heuristic: If numbering is sequential within column → column-first; if alternates → row-first
        - Sort text blocks: `sorted(blocks, key=lambda b: (b['column'], b['top']))` for column-first
      
      - **Validation**:
        - Check reference numbering continuity: [1], [2], [3], ... (no gaps)
        - If gaps detected: Possible reading order error → retry with alternative strategy
        - Flag pages with non-standard layouts (>2 columns, mixed layouts)
   
   d. **Text Cleaning and Normalization**:
      
      - **Encoding Fixes**:
        - Detect encoding: Use `chardet` library
        - Convert to UTF-8: `text.encode('latin-1').decode('utf-8', errors='ignore')`
        - Fix common corruptions: "\xef\xbf\xbd" (replacement char) → log warning
      
      - **Ligature Expansion**:
        - Replace: "ﬁ" → "fi", "ﬂ" → "fl", "ﬀ" → "ff", "ﬃ" → "ffi", "ﬄ" → "ffl"
        - Common in older PDFs with certain fonts
      
      - **Hyphenation Handling**:
        - Detect end-of-line hyphens: `word-\n`
        - Dehyphenate: `"atten-\ntion" → "attention"`
        - Preserve intentional hyphens: "state-of-the-art"
      
      - **Whitespace Normalization**:
        - Replace multiple spaces with single space
        - Normalize line breaks: `\r\n → \n`
        - Remove trailing/leading whitespace
      
      - **Special Character Handling**:
        - Preserve: DOIs (10.xxxx/...), URLs (http://...), emails
        - Normalize quotes: "smart quotes" → " (straight quotes)
        - Handle mathematical symbols: Preserve if in title (e.g., "α-diversity")
   
   e. **Preprocessing Output**:
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

2. **Parsing with GROBID**:
   
   **Purpose**: Extract structured bibliographic metadata from raw reference text using GROBID's machine learning models.
   
   a. **GROBID Setup and Configuration**:
      
      - **Installation**:
        - Docker (recommended): `docker pull lfoppiano/grobid:0.7.3`
        - Run: `docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3`
        - Verify: `curl http://localhost:8070/api/isalive` (should return "true")
      
      - **Configuration** (`config/grobid.yaml`):
        ```yaml
        grobid:
          consolidation:
            enabled: true  # Use CrossRef API for metadata enrichment
            crossref_email: "your@email.com"
            crossref_threads: 2
          models:
            citation: "citation-model.wapiti"  # Default model
            referenceSegmenter: "reference-segmenter-model.wapiti"
          batch_size: 25  # Process multiple references at once
          timeout: 60  # Seconds per request
        ```
      
      - **API Endpoints**:
        - `/api/processReferences`: Parse list of raw reference strings
        - `/api/processCitationList`: Parse references from full text
        - `/api/processHeaderDocument`: Parse document header (for venue detection)
      
      - **Performance Tuning**:
        - Increase JVM heap: `-Xmx4G` for large batches
        - Enable multi-threading: `concurrency: 10` in config
        - Use batch processing: Send 25-50 references per request
   
   b. **Input Preparation**:
      
      - **Format Requirements**:
        - One reference per line, or TEI XML wrapper
        - Max length: 5000 chars per reference (split if longer)
        - Encoding: UTF-8
      
      - **Pre-parsing Cleanup**:
        - Remove residual headers/footers that leaked into reference section
        - Fix obvious errors: Double periods "..", missing spaces "Smith,J."
        - Normalize bullets/numbering: Strip `[1]`, `1.`, etc. (GROBID handles unnumbered better)
      
      - **Batching Strategy**:
        - Group references in batches of 25-50
        - Parallel processing: Send batches to multiple GROBID instances (if available)
        - Respect rate limits: Max 10 req/sec for single instance
   
   c. **GROBID API Call**:
      
      ```python
      import requests
      
      def parse_references_grobid(references_text, consolidate=True):
          url = "http://localhost:8070/api/processReferences"
          params = {
              "input": references_text,
              "consolidateCitations": 1 if consolidate else 0,
              "includeRawCitations": 1
          }
          response = requests.post(url, data=params, timeout=60)
          
          if response.status_code == 200:
              return response.text  # TEI XML format
          elif response.status_code == 503:
              # Service unavailable, retry with backoff
              time.sleep(2)
              return parse_references_grobid(references_text, consolidate)
          else:
              raise Exception(f"GROBID error: {response.status_code}")
      ```
   
   d. **TEI XML Parsing**:
      
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
      
      - **Extraction Logic** (using `xml.etree.ElementTree` or `lxml`):
        ```python
        from lxml import etree
        
        def parse_tei_xml(tei_xml):
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            root = etree.fromstring(tei_xml.encode('utf-8'))
            
            references = []
            for bib_struct in root.xpath('//tei:biblStruct', namespaces=namespaces):
                ref = {}
                
                # Title
                title_elem = bib_struct.xpath('.//tei:title[@level="a" or @level="m"]', namespaces=namespaces)
                ref['title'] = title_elem[0].text if title_elem else None
                
                # Authors
                authors = []
                for author in bib_struct.xpath('.//tei:author/tei:persName', namespaces=namespaces):
                    forename = author.xpath('.//tei:forename', namespaces=namespaces)
                    surname = author.xpath('.//tei:surname', namespaces=namespaces)
                    authors.append({
                        'first': forename[0].text if forename else None,
                        'last': surname[0].text if surname else None
                    })
                ref['authors'] = authors
                
                # Year
                date_elem = bib_struct.xpath('.//tei:date[@type="published"]/@when', namespaces=namespaces)
                ref['year'] = int(date_elem[0][:4]) if date_elem else None
                
                # Venue
                venue_elem = bib_struct.xpath('.//tei:title[@level="j" or @level="m"]', namespaces=namespaces)
                ref['venue'] = {'raw': venue_elem[0].text if venue_elem else None}
                
                # Identifiers
                doi = bib_struct.xpath('.//tei:idno[@type="DOI"]', namespaces=namespaces)
                arxiv = bib_struct.xpath('.//tei:idno[@type="arXiv"]', namespaces=namespaces)
                ref['identifiers'] = {
                    'doi': doi[0].text if doi else None,
                    'arxiv_id': arxiv[0].text if arxiv else None
                }
                
                references.append(ref)
            
            return references
        ```
      
      - **Confidence Scores**: GROBID provides per-field confidence in XML attributes
        - Extract: `<title coords="..." conf="0.92">Title</title>`
        - Aggregate: `extraction_confidence = mean([field_confidences])`
   
   e. **Optional Routing for Venue-Specific Formats**:
      
      **Purpose**: Improve accuracy for papers from specific venues with unique citation styles.
      
      - **Venue Detection** (from paper metadata):
        
        1. **From PDF Metadata**:
           - Extract using PyMuPDF: `doc.metadata['subject']` or `doc.metadata['keywords']`
           - Parse first page: Look for conference/journal name in header/footer
           - Use GROBID's `/api/processHeaderDocument`: Extracts title, authors, venue from first page
        
        2. **From DOI Lookup** (if paper DOI available):
           - Query CrossRef: `https://api.crossref.org/works/{doi}`
           - Extract `container-title`: Journal/conference name
        
        3. **From Filename/Path** (if follows convention):
           - Parse: `"ACM_CHI_2023_paper.pdf" → venue="ACM CHI", year=2023`
        
        4. **ML-Based Classification** (optional):
           - Train classifier on first page text features (n-grams, keywords)
           - Classes: ["ACM", "IEEE", "Springer", "Elsevier", "arXiv", "NeurIPS", "ICML", "ACL", "Other"]
           - Use FastText or BERT-based classifier
      
      - **Venue-Specific Models** (custom GROBID training):
        
        - **Training Process**:
          1. Collect 200+ papers from target venue
          2. Manually annotate references in GROBID XML format
          3. Train venue-specific model: `./gradlew train_citation -PgrobidHome=...`
          4. Deploy: Replace default model for detected venue
        
        - **Model Selection Logic**:
          ```python
          def select_grobid_model(venue):
              venue_models = {
                  "ACM": "models/citation-acm.wapiti",
                  "IEEE": "models/citation-ieee.wapiti",
                  "NeurIPS": "models/citation-neurips.wapiti",
                  # ... more venues
              }
              return venue_models.get(venue, "models/citation-default.wapiti")
          ```
      
      - **Post-Processing Rules** (venue-specific adjustments):
        
        - **ACM Style**:
          - Fix: "Proc. ACM" abbreviations → expand to full conference name
          - Regex: `Proc\. ACM (\w+)` → lookup in ACM conference database
          - Handle DOI format: `10.1145/...` (always ACM)
        
        - **IEEE Style**:
          - Fix: Journal abbreviations ("Trans. Pattern Anal. Mach. Intell." → "IEEE Transactions on PAMI")
          - Volume/issue parsing: IEEE uses `vol. X, no. Y, pp. Z-W` format
        
        - **arXiv Style**:
          - Identify preprints: Look for "arXiv:XXXX.XXXXX"
          - Dual attribution: Some have both arXiv ID and conference venue (published version)
          - Prioritize: Conference venue over arXiv if both present
        
        - **BioRxiv/MedRxiv**:
          - DOI pattern: `10.1101/...` (bioRxiv/medRxiv specific)
          - Often missing traditional venue: Mark as preprint
      
      - **Routing Output Metadata**:
        ```json
        {
          "venue_routing": {
            "detected_venue": "NeurIPS",
            "confidence": 0.88,
            "detection_method": "header_parsing",
            "model_used": "citation-neurips.wapiti",
            "post_processing_rules_applied": ["venue_abbreviation_expansion"]
          }
        }
        ```
   
   f. **Consolidation** (Metadata Enrichment via CrossRef):
      
      - **When Enabled**: GROBID queries CrossRef API to validate/enrich metadata
      - **Process**:
        1. GROBID extracts title + authors + year
        2. Queries CrossRef: `https://api.crossref.org/works?query.bibliographic={title}+{authors}`
        3. If match found (score >0.8), enriches: Adds DOI, full venue name, volume/issue
      - **Trade-offs**:
        - Pros: Higher accuracy (DOI retrieved), complete metadata
        - Cons: Slower (200-500ms per reference), requires internet, API rate limits
      - **Configuration**: Enable for critical applications, disable for speed
      - **Caching**: Store consolidated results to avoid redundant API calls
   
   g. **Error Handling and Quality Control**:
      
      - **Parsing Failures**:
        
        1. **GROBID Timeout**:
           - Cause: Complex reference, long text
           - Solution: Retry with shorter text (truncate to 2000 chars), or skip consolidation
        
        2. **Invalid TEI XML**:
           - Cause: GROBID bug, malformed input
           - Solution: Validate XML with schema, attempt lenient parsing with `recover=True`
        
        3. **Empty Output**:
           - Cause: Reference format unrecognized (e.g., web page, dataset)
           - Solution: Flag as `extraction_failed`, attempt rule-based fallback
        
        4. **Partial Parsing**:
           - Cause: Missing fields (e.g., no year extracted)
           - Solution: Accept partial data, flag low confidence
      
      - **Rule-Based Fallback** (for GROBID failures):
        
        - **Author Extraction**:
          - Regex: `(?P<authors>[A-Z][a-z]+(?:,\s*[A-Z]\.)?(?:,\s*[A-Z][a-z]+(?:,\s*[A-Z]\.)?)*(?:,?\s*(?:and|&)\s*[A-Z][a-z]+)?)`
          - Example: "Smith, J., Doe, A., and Lee, K." → ["Smith, J.", "Doe, A.", "Lee, K."]
        
        - **Year Extraction**:
          - Regex: `\b(19|20)\d{2}\b` (find 4-digit year)
          - Validate: 1900 ≤ year ≤ current_year + 2
        
        - **Title Extraction** (heuristic):
          - Assume: Title is longest quoted string, or text between authors and year
          - Pattern: After authors, before year or venue, often in quotes or italics
        
        - **DOI Extraction**:
          - Regex: `10\.\d{4,9}/[-._;()/:A-Z0-9]+`
          - Validate: Check format, remove trailing punctuation
        
        - **Confidence**: Flag fallback extractions with `extraction_method="regex_fallback"`, confidence = 0.5-0.7
      
      - **Quality Metrics**:
        
        - **Completeness Score**: `(num_non_null_fields / total_expected_fields)`
          - Essential fields: title, authors, year (weight: 0.8)
          - Optional fields: venue, identifiers (weight: 0.2)
          - Threshold: Completeness <0.5 → flag as `low_quality`
        
        - **Confidence Score**: Aggregate GROBID's per-field confidences
          - Formula: `extraction_confidence = (Σ field_conf * field_weight) / Σ field_weight`
          - Weights: title (0.4), authors (0.3), year (0.2), venue (0.1)
          - Threshold: Confidence <0.6 → flag as `uncertain`
        
        - **Format Validity**: Check field formats
          - Year: Is integer, in valid range
          - DOI: Matches DOI regex
          - Authors: Has at least one author
          - Score: `format_validity = (num_valid_fields / num_present_fields)`
      
      - **Error Logging**:
        ```json
        {
          "errors": [
            {
              "reference_id": "ref_003",
              "error_type": "grobid_timeout",
              "error_message": "GROBID request timeout after 60s",
              "fallback_used": "regex_extraction",
              "extraction_confidence": 0.65
            }
          ]
        }
        ```

3. **Output Formatting**:
   
   **Purpose**: Structure extracted references into consistent, machine-readable JSON format with comprehensive metadata.
   
   a. **JSON Schema Definition**:
      
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
            "model": "citation-default.wapiti",
            "consolidation_enabled": true,
            "venue_routing_applied": false
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
            "reference_index": 1,
            "raw_text": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).",
            
            "parsed_data": {
              "title": "Attention is all you need",
              "title_original": "Attention is all you need",
              
              "authors": [
                {"first": "Ashish", "middle": null, "last": "Vaswani", "full": "Ashish Vaswani"},
                {"first": "Noam", "middle": null, "last": "Shazeer", "full": "Noam Shazeer"},
                {"first": "Niki", "middle": null, "last": "Parmar", "full": "Niki Parmar"},
                {"first": "Jakob", "middle": null, "last": "Uszkoreit", "full": "Jakob Uszkoreit"},
                {"first": "Llion", "middle": null, "last": "Jones", "full": "Llion Jones"},
                {"first": "Aidan N.", "middle": null, "last": "Gomez", "full": "Aidan N. Gomez"},
                {"first": "Łukasz", "middle": null, "last": "Kaiser", "full": "Łukasz Kaiser"},
                {"first": "Illia", "middle": null, "last": "Polosukhin", "full": "Illia Polosukhin"}
              ],
              "authors_truncated": false,
              
              "year": 2017,
              "year_raw": "2017",
              
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
            },
            
            "metadata": {
              "grobid_consolidation": {
                "used": true,
                "successful": true,
                "source": "crossref",
                "fields_enriched": ["doi", "venue.normalized"]
              },
              "venue_routing": null,
              "fallback_parsing": false
            }
          }
          // Additional references...
        ]
      }
      ```
   
   b. **Completeness Validation**:
      
      - **Essential Fields Check**:
        - Title: Required (non-null, length >5 chars)
        - Authors: Required (at least 1 author)
        - Year: Required (valid year)
        - Action if missing: Flag as `incomplete`, attempt extraction from raw text
      
      - **Optional Fields Enhancement**:
        
        1. **DOI Extraction** (if missing):
           - Regex: `10\.\d{4,9}/[-._;()/:A-Z0-9]+` on raw text
           - Validation: Query `https://doi.org/{doi}` (follow redirect, check 200 status)
           - If found: Add to `identifiers.doi`, increase confidence by 0.05
        
        2. **arXiv ID Extraction** (if missing):
           - Regex: `arXiv:(\d{4}\.\d{4,5}(v\d+)?)|([a-z-]+/\d{7})`
           - Normalize: Remove "arXiv:" prefix, store version separately
           - Validate: Query `http://export.arxiv.org/api/query?id_list={arxiv_id}`
        
        3. **URL Extraction** (if missing):
           - Regex: `https?://[^\s<>"]+[^\s<>".,;:)]`
           - Filter: Remove URLs ending in image extensions (.jpg, .png)
           - Store: Up to 1 most relevant URL (prefer doi.org, arxiv.org)
        
        4. **Venue Type Classification** (if not detected):
           - Keywords: "Proceedings"/"Conference" → conference
           - Keywords: "Journal"/"Transactions" → journal
           - Keywords: "arXiv"/"bioRxiv" → preprint
           - Keywords: "Workshop" → workshop
           - Default: "other"
        
        5. **Pages Normalization** (if present):
           - Format: "123-145" or "pp. 123-145" or "pages 123 to 145"
           - Extract: `(\d+)\s*[-–—to]+\s*(\d+)` → "123-145"
           - Validate: end_page > start_page
      
      - **Cross-Field Consistency Checks**:
        
        1. **Author-Year Mismatch**:
           - If raw text has (AuthorYear) pattern, check extracted year matches
           - Example: Raw has "(Vaswani2017)" but year extracted as 2016 → flag warning
        
        2. **Title-Venue Mismatch**:
           - If title contains venue abbreviation (e.g., "NeurIPS" in title), check consistency
           - Usually error: Title shouldn't contain venue
        
        3. **DOI-Metadata Consistency** (if DOI exists):
           - Query DOI to get metadata, compare with extracted data
           - If major mismatch (e.g., different year) → flag warning
        
        4. **Duplicate Detection**:
           - Compare current reference with previous ones (title similarity)
           - If title Levenshtein >0.95 with previous → flag as potential duplicate
   
   c. **Quality Scoring and Flagging**:
      
      - **Completeness Score**:
        ```python
        weights = {"title": 0.3, "authors": 0.3, "year": 0.2, "venue": 0.1, "identifiers": 0.1}
        completeness = sum(
            weights[field] for field in weights 
            if parsed_data[field] is not None and parsed_data[field] != ""
        )
        ```
      
      - **Format Validity Score**:
        ```python
        checks = [
            year is not None and 1900 <= year <= 2028,
            authors is not None and len(authors) > 0,
            title is not None and len(title) > 5,
            doi is None or doi_regex.match(doi),
            arxiv_id is None or arxiv_regex.match(arxiv_id)
        ]
        format_validity = sum(1 for c in checks if c) / len(checks)
        ```
      
      - **Overall Extraction Confidence**:
        ```python
        extraction_confidence = (
            0.5 * grobid_confidence +
            0.3 * completeness_score +
            0.2 * format_validity_score
        )
        ```
      
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
   
   d. **Error Handling and Recovery**:
      
      - **Malformed JSON Prevention**:
        - Escape special chars in strings: `", \, \n, \t`
        - Handle null values: Use `null` (not `None` or empty string)
        - Validate JSON before returning: `json.loads(json.dumps(data))`
      
      - **Encoding Issues**:
        - Force UTF-8: `json.dumps(data, ensure_ascii=False)`
        - Handle non-printable chars: Replace with Unicode escape sequences
      
      - **Large Fields Truncation**:
        - If title >500 chars: Likely parsing error, truncate and flag
        - If raw_text >5000 chars: Truncate for storage
      
      - **Missing Reference Handling**:
        - If extraction completely fails: Create minimal entry with raw_text only
        ```json
        {
          "id": "X",
          "raw_text": "...",
          "parsed_data": null,
          "extraction_quality": {
            "extraction_confidence": 0.0,
            "extraction_method": "failed",
            "issues": ["extraction_failed"]
          }
        }
        ```
   
   e. **Output Variants**:
      
      1. **Full Output** (default):
         - All fields, metadata, quality scores
         - Use: For verification stage, detailed analysis
      
      2. **Compact Output**:
         - Only parsed_data, exclude metadata/quality
         - Use: For storage efficiency, simple applications
      
      3. **Extended Output** (with venue routing):
         - Add `venue_specific_rules_applied: true`
         - Include routing metadata: detected venue, model used
      
      4. **Export Formats**:
         - **JSON** (primary): Structured, machine-readable
         - **CSV**: Flattened (one row per reference)
         - **BibTeX**: For citation managers
         - **RIS**: For reference management software
         - **TEI XML**: For archival/interoperability
   
   f. **Final Validation**:
      
      - **Schema Compliance**: Validate against JSON Schema
      - **Referential Integrity**: Check all reference IDs are unique
      - **Summary Statistics**: Count parsed/failed/partial references
      - **Quality Thresholds**:
        - Average extraction_confidence >0.75 → acceptable
        - Failure rate <10% → acceptable
        - If below thresholds: Flag entire batch for review

### Extraction Benchmark

**Purpose**: Rigorously evaluate the extraction system's accuracy, robustness, and performance across diverse reference formats and quality levels.

To evaluate extraction quality, we adopt and extend GROBID's benchmarking framework [src](https://grobid.readthedocs.io/en/latest/benchmarks/Benchmarking-biorxiv/), focusing on field-level and overall metrics with comprehensive error analysis.

#### 1. **Dataset Construction**

**Ground Truth Sources** (Multi-source with manual validation):

a. **Gold Standard Datasets**:
   - **GROBID Test Set**: 500+ references with TEI annotations
   - **PubMed Central**: 1000+ biomedical references with structured XML
   - **arXiv Papers**: 800+ preprints with author-provided metadata
   - **DBLP**: 600+ computer science references (manually curated)
   - **ACL Anthology**: 400+ NLP papers with clean bibliography

b. **Manual Annotation Protocol**:
   - **Annotators**: 3+ domain experts per paper
   - **Inter-annotator Agreement**: 
     - Compute Cohen's κ for categorical fields (venue type)
     - Compute Pearson correlation for continuous fields (confidence)
     - Requirement: κ > 0.80 or correlation > 0.85
   - **Adjudication**: Disagreements resolved by senior annotator
   - **Annotation Schema**: Field-level ground truth + expected verdict

**Dataset Composition** (Stratified sampling for comprehensive coverage):

a. **By Venue Type** (test diversity):
   - Conferences: 35% (e.g., NeurIPS, ICML, ACL, CHI)
   - Journals: 30% (e.g., Nature, Science, TPAMI, JMLR)
   - Preprints: 15% (arXiv, bioRxiv, medRxiv)
   - Books/Theses: 10% (monographs, dissertations)
   - Other: 10% (tech reports, datasets, websites, software)

b. **By Citation Style** (test robustness to format variations):
   - Numbered: 40% (e.g., `[1] Author...`)
   - Author-year: 30% (e.g., `Smith (2020)...`)
   - IEEE: 15% (e.g., `[1] A. Smith, "Title,"...`)
   - APA: 10% (e.g., `Smith, A. (2020). Title...`)
   - Other/Mixed: 5%

c. **By Quality Level** (test degradation handling):
   - **Clean** (60%): Perfect formatting, no OCR errors
   - **OCR Errors** (15%): Scanned papers, character substitutions (simulate: 'l' → '1', 'o' → '0')
   - **Formatting Issues** (15%): Non-standard styles, missing fields, inconsistent spacing
   - **Truncated** (5%): Incomplete references (cut off at page boundary)
   - **Corrupted** (5%): Severe errors, garbled text

d. **By Era** (test temporal coverage):
   - Recent (2020-2026): 40%
   - Modern (2010-2019): 35%
   - Historical (2000-2009): 15%
   - Legacy (pre-2000): 10%

e. **By Language** (test multilingual support):
   - English: 85%
   - Non-English with Latin script (French, German, Spanish): 10%
   - Non-Latin scripts (Chinese, Japanese, Russian): 5%

f. **Edge Cases** (stress testing) - 300+ references:
   - Author name variations: 50 (Jr., Sr., III, hyphenated, single-name)
   - Special characters in titles: 40 (math symbols, Greek letters, diacritics)
   - Venue name ambiguity: 40 (abbreviations, name changes)
   - Multiple identifiers: 30 (both DOI and arXiv)
   - Missing fields: 40 (no year, no venue, no authors)
   - Non-traditional formats: 50 (datasets, software, URLs)
   - Very long references: 20 (>500 chars, multiple works cited together)
   - Duplicate/near-duplicate references: 30

**Data Split** (ensure no leakage):
- **Training Set**: 60% (for venue-specific model training, threshold tuning)
- **Validation Set**: 20% (for hyperparameter selection, model selection)
- **Test Set**: 20% (final evaluation, never seen during development)
- **Stratification**: Maintain venue/quality/style distributions across splits
- **Deduplication**: Same paper cannot appear in multiple splits

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

**Purpose**: Define how extracted fields are compared with ground truth to determine correctness.

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
   
   - **Definitions**:
     - TP (True Positive): Field extracted correctly (matches ground truth)
     - FP (False Positive): Field extracted incorrectly (doesn't match ground truth)
     - FN (False Negative): Field missing in extraction (present in ground truth)
     - TN (True Negative): Field correctly not extracted (not present in ground truth)
   
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

b. **Per-Field Reporting**:
   
   | Field        | Precision | Recall | F1    | Accuracy | Matches | Mismatches | Missing |
   |--------------|-----------|--------|-------|----------|---------|------------|---------|
   | **Title**    | 0.94      | 0.96   | 0.95  | 0.93     | 2850    | 120        | 30      |
   | **Authors**  | 0.88      | 0.91   | 0.89  | 0.86     | 2730    | 180        | 90      |
   | **Year**     | 0.99      | 0.99   | 0.99  | 0.99     | 2970    | 15         | 15      |
   | **Venue**    | 0.86      | 0.89   | 0.87  | 0.84     | 2670    | 210        | 120     |
   | **DOI**      | 0.92      | 0.85   | 0.88  | 0.91     | 1700    | 80         | 220     |
   | **arXiv ID** | 0.95      | 0.88   | 0.91  | 0.94     | 440     | 20         | 40      |

c. **Matching Method Comparison**:
   
   | Field   | Strict | Soft  | Levenshtein (≥0.8) | Ratcliff (≥0.95) |
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

**Overall Metrics** (Reference-level evaluation):

a. **Reference Correctness**:
   - **Fully Correct**: All essential fields (title, authors, year) match ground truth
   - **Partially Correct**: ≥2 of 3 essential fields match
   - **Incorrect**: <2 essential fields match or extraction failed
   - **Accuracy**: `Fully Correct / Total References`

b. **Extraction Success Rate**:
   - **Success**: At least title OR (authors + year) extracted
   - **Partial**: Some fields extracted but incomplete
   - **Failure**: No fields extracted or all fields incorrect
   - **Rate**: `(Success + Partial) / Total`

**Error Analysis Framework**:

a. **Error Type Taxonomy**:
   1. **Field Missing**: Ground truth has field, extraction doesn't
   2. **Field Hallucination**: Extraction has field, ground truth doesn't (or wrong)
   3. **Field Incorrect**: Both have field but values don't match
   4. **Boundary Error**: Field value bleeds into adjacent field
   5. **Format Error**: Field extracted but wrong format (e.g., year as "2017-2018")
   6. **Parsing Failure**: GROBID/parser completely failed

b. **Per-Error-Type Analysis**:
   ```
   Error Type             | Count | % of Errors | Example
   -----------------------|-------|-------------|---------------------------
   Field Missing (Title)  | 35    | 12%         | Title not extracted
   Field Missing (Year)   | 15    | 5%          | Year not found in raw text
   Field Incorrect (Venue)| 85    | 29%         | "NIPS" vs "NeurIPS"
   Boundary Error         | 42    | 14%         | Year included in title
   Format Error (Authors) | 28    | 9%          | "et al." not handled
   Parsing Failure        | 18    | 6%          | GROBID timeout
   Other                  | 77    | 25%         | Various issues
   -----------------------|-------|-------------|---------------------------
   Total Errors           | 300   | 100%        |
   ```

c. **Quality Stratification**:
   - Report metrics separately for each quality level:
   ```
   Quality Level | F1    | Recall | Precision | Notes
   --------------|-------|--------|-----------|----------------------
   Clean         | 0.95  | 0.96   | 0.94      | Baseline performance
   OCR Errors    | 0.82  | 0.80   | 0.84      | -13% F1 degradation
   Formatting    | 0.78  | 0.75   | 0.81      | -17% F1 degradation
   Truncated     | 0.65  | 0.62   | 0.68      | -30% F1 (severe)
   Corrupted     | 0.52  | 0.48   | 0.56      | -43% F1 (critical)
   ```

#### 4. **Benchmarking Protocols**

**Baseline Comparisons**:

a. **Naive Baselines**:
   1. **Regex-Only**: Hand-crafted regex patterns
   2. **Simple Heuristics**: Author-year pattern matching
   3. **No Preprocessing**: Direct GROBID without PDF cleanup

b. **GROBID Variants**:
   1. **GROBID Default**: Out-of-box model
   2. **GROBID + Consolidation**: With CrossRef enrichment
   3. **GROBID Venue-Specific**: Custom models per venue

c. **External Systems** (if accessible):
   1. **ParsCit**: Reference string parsing
   2. **CERMINE**: Full-text extraction + reference parsing
   3. **Science Parse** (Allen AI)
   4. **anystyle.io**: ML-based reference parsing

**Ablation Studies** (Measure component contributions):

1. **Without OCR Fallback**: How much does OCR help?
2. **Without Column Detection**: Impact of layout analysis
3. **Without Consolidation**: CrossRef enrichment value
4. **Without Venue Routing**: Venue-specific model benefit
5. **Without Fallback Parsing**: GROBID-only vs. hybrid

**Statistical Significance Testing**:
- **Paired t-test**: Compare F1 scores of proposed vs. baseline on same test set
- **McNemar's test**: For binary outcomes (correct/incorrect)
- **Bootstrap resampling**: Compute 95% confidence intervals (1000 iterations)
- **Report**: p-value, effect size (Cohen's d), confidence intervals
- **Significance threshold**: p < 0.05 (with Bonferroni correction for multiple comparisons)

**Reporting Standards**:

1. **Main Results Table**:
   ```
   System                  | Macro F1 | Weighted F1 | Title F1 | Author F1 | Year Acc |
   ------------------------|----------|-------------|----------|-----------|----------|
   Proposed (Full)         | 0.91***  | 0.92***     | 0.95     | 0.89      | 0.99     |
   - w/o Venue Routing     | 0.89**   | 0.90**      | 0.95     | 0.88      | 0.99     |
   - w/o Consolidation     | 0.88**   | 0.89**      | 0.93     | 0.87      | 0.99     |
   - w/o OCR Fallback      | 0.85*    | 0.87*       | 0.91     | 0.84      | 0.98     |
   GROBID Default          | 0.87     | 0.88        | 0.93     | 0.86      | 0.99     |
   ParsCit                 | 0.84     | 0.85        | 0.90     | 0.83      | 0.97     |
   Regex-Only Baseline     | 0.72     | 0.74        | 0.85     | 0.75      | 0.95     |
   
   *** p < 0.001, ** p < 0.01, * p < 0.05 (vs. GROBID Default)
   ```

2. **Confusion Matrix** (for overall correctness):\n   ```\n                     | Predicted:        | Predicted:         | Predicted:  |\n                     | Fully Correct     | Partially Correct  | Incorrect   |\n   ------------------|-------------------|-----------------------|-------------|\n   GT: Fully Correct | 2650 (TP)         | 120 (FP)              | 30 (FP)     |\n   GT: Partial       | 85 (FN)           | 180 (TP)              | 35 (FP)     |\n   GT: Incorrect     | 15 (FN)           | 25 (FN)               | 60 (TP)     |\n   ```

3. **Error Analysis Summary**:\n   - Top-3 error types with mitigation strategies\n   - Examples of each error type with explanation\n   - Failure mode analysis: When does system fail most?\n\n4. **Quality Degradation Curves**:\n   - Plot: F1 score vs. OCR error rate (0%, 5%, 10%, 20%)\n   - Plot: F1 score vs. reference complexity (measured by length, fields present)\n   - Plot: Processing time vs. reference count (scalability)\n\n5. **Computational Performance**:\n   ```\n   Metric                        | Value         | Notes\n   ------------------------------|---------------|-------------------------\n   Avg. time per reference       | 125ms         | Including GROBID call\n   Throughput                    | 480 refs/min  | With parallelization\n   GROBID API latency           | 80ms (p50)    | Bottleneck identified\n   Preprocessing overhead        | 30ms          | PDF extraction + cleanup\n   Peak memory usage             | 2.1GB         | For 1000 references\n   ```

6. **Dataset Statistics**:\n   - Total references: 3000\n   - Train/Val/Test: 1800/600/600\n   - Venue distribution, quality distribution, style distribution
   - Average reference length: 245 chars
   - Average fields per reference: 4.8

#### 5. **Continuous Benchmarking & Monitoring**

**Version Control**:\n- Track metrics across system versions\n- Detect performance regressions: Alert if F1 drops >2% between versions\n- Maintain performance dashboard: Real-time monitoring of extraction quality

**Production Monitoring** (if deployed):\n- Log extraction confidence for all references\n- Alert on low-confidence extractions (<0.6)\n- Sample random references for manual review (QA)\n- Track error rates by venue type, era, quality\n- Monthly re-evaluation on held-out test set

**Dataset Expansion**:\n- Continuously add new references: As new paper formats emerge\n- Quarterly dataset updates: Incorporate edge cases from production\n- Community contributions: Accept annotated references from users\n- Target: Grow to 10,000+ references over time

**Reproducibility Checklist**:\n- [ ] Code repository public (GitHub) with Apache 2.0 license\n- [ ] Dataset available (Zenodo/Figshare) with CC-BY license\n- [ ] GROBID version documented (0.7.3)\n- [ ] Python environment documented (requirements.txt, Python 3.9+)\n- [ ] Random seeds fixed (for shuffling, model initialization)\n- [ ] Hardware specifications documented (CPU, RAM, GPU if used)\n- [ ] Execution instructions (README with step-by-step guide)\n- [ ] Pre-trained models available (if venue-specific routing used)\n- [ ] Example outputs provided (sample JSON for inspection)

## Verification Stage

This stage verifies extracted references against external authoritative sources (e.g., CrossRef, Semantic Scholar, arXiv API) to detect inaccuracies, fabrications, or incompleteness. It involves normalization to handle variations, fetching canonical metadata, multi-metric similarity computation, and verdict assignment.

**Detailed Workflow:**
1. **Normalization**:
   
   **Purpose**: Standardize extracted reference fields to canonical forms for consistent comparison with external metadata.
   
   **Pre-processing Validation**:
   - Check for null/empty fields and log warnings
   - Detect encoding issues (UTF-8, Latin-1) and normalize to UTF-8
   - Remove zero-width characters and control characters (regex: `[\x00-\x1F\x7F-\x9F]`)
   - Trim excessive whitespace (replace `\s+` with single space)
   
   **Field-Specific Normalization**:
   
   a. **Authors** (Multi-stage pipeline):
      - **Splitting**: Parse author string using delimiters (`;`, `,`, `and`, `&`)
      - **Name Parsing**: Use `nameparser` library or custom regex:
        - Extract first name, middle initial, last name (handle suffixes like Jr., Sr., III)
        - Pattern: `(?P<first>[A-Z][a-z]+)\s+(?P<middle>[A-Z]\.)?\s*(?P<last>[A-Z][a-z]+)`
      - **Abbreviation Expansion**: 
        - Build lookup table from DBLP/OpenAlex author aliases
        - Apply heuristics: "A. Vaswani" → "Ashish Vaswani" if unambiguous in context
        - Fallback: Keep abbreviated form if multiple expansions possible
      - **Affiliation Removal**: Strip parenthetical/bracketed affiliations using regex: `\([^)]*\)|\[[^\]]*\]`
      - **Special Character Normalization**: Convert diacritics (e.g., "Müller" → "Muller") using `unidecode` library
      - **Ordering**: Preserve original order but store as list of dicts: `[{"first": "Ashish", "last": "Vaswani"}, ...]`
      - **Et al. Handling**: If "et al." detected, flag `authors_truncated=True` and store partial list
      - **Edge Cases**: 
        - Single-name authors (e.g., "Plato"): Store as last name
        - Corporate authors (e.g., "WHO"): Flag with `is_corporate=True`
        - Non-Latin scripts: Transliterate using `transliterate` library (Cyrillic, CJK)
   
   b. **Title** (Preserve original + normalized versions):
      - **Store Original**: Keep `title_original` for display
      - **Case Normalization**: Convert to lowercase for comparison
      - **Punctuation Handling**: 
        - Remove most punctuation but preserve hyphens in compound words
        - Remove colons/subtitles optionally (store as `title_main` and `title_subtitle`)
      - **Stop Word Removal**: Optionally remove common words ("the", "a", "an") for fuzzy matching
      - **Stemming/Lemmatization**: Apply NLTK Porter Stemmer or spaCy lemmatizer
        - Store both stemmed and lemmatized versions for multi-strategy matching
      - **Mathematical Expressions**: Preserve LaTeX/MathML (regex: `\$[^$]+\$`) or convert to plain text
      - **Acronyms**: Normalize spacing (e.g., "N L P" → "NLP")
      - **Semantic Embedding**: Generate SciBERT/Sentence-BERT embedding (768-dim vector)
        - Batch process for efficiency (batch size: 32)
        - Store embeddings in vector database (e.g., FAISS, Pinecone) for retrieval
   
   c. **Year** (Validation + extraction):
      - **Type Conversion**: Parse string to integer, handle exceptions
      - **Range Handling**: Extract first year from ranges ("2017-2018" → 2017) or average
      - **Validation**: Check range (1900 ≤ year ≤ current_year + 2 for preprints)
      - **Format Detection**: Handle formats like "'17" → 2017 (assume 20xx unless >current_year)
      - **Missing Year**: If null, attempt extraction from raw text using regex: `\b(19|20)\d{2}\b`
      - **Edge Cases**: 
        - Forthcoming papers: Flag as `year_uncertain=True`
        - Historical papers (pre-1900): Allow if validated against external source
   
   d. **Venue** (Hierarchical normalization):
      - **Abbreviation Expansion**: 
        - Use lookup table (JSON/CSV) mapping abbreviations to full names
        - Example mappings: {"NIPS": "NeurIPS", "ICML": "International Conference on Machine Learning"}
        - Source: Scrape from DBLP, WikiCFP, or maintain custom list
      - **Type Classification**: 
        - Detect venue type using keywords: "Proceedings"/"Conference" → conference, "Journal" → journal, "arXiv" → preprint
        - Store as `venue_type` enum: ["conference", "journal", "workshop", "preprint", "book", "thesis", "other"]
      - **Acronym Extraction**: Extract conference acronym (e.g., "ACL 2023" from "Proceedings of ACL")
      - **Normalization**: Lowercase, remove "Proceedings of", "Journal of", etc.
      - **External Validation**: Query Wikidata/DBLP for canonical venue name
      - **Ranking/Impact**: Optionally fetch venue rank (e.g., CORE rank, h-index) for filtering
   
   e. **Identifiers** (Validation + extraction):
      - **DOI**: 
        - Validate format using regex: `10\.\d{4,9}/[-._;()/:A-Z0-9]+`
        - Verify checksum if DOI uses check digit
        - Normalize: Remove "https://doi.org/" prefix, lowercase
        - Query DOI.org API to validate existence (with caching)
      - **arXiv ID**: 
        - Validate format: Old (`math/0601001`) vs. new (`1706.03762`, `2301.12345`)
        - Regex: `(\d{4}\.\d{4,5}(v\d+)?)|([a-z-]+/\d{7})`
        - Normalize: Remove version suffix for primary ID, store version separately
      - **ISBN**: 
        - Validate ISBN-10/ISBN-13 checksum
        - Convert ISBN-10 to ISBN-13 for uniformity
      - **PMID/PMC ID**: Validate format for biomedical papers
      - **Extraction from Raw Text**: If identifiers missing, scan raw reference string:
        - DOI: Apply regex to raw text
        - arXiv: Look for "arXiv:" prefix
   
   **Post-Normalization Quality Checks**:
   - Flag references with >50% missing fields as `low_quality`
   - Compute normalization confidence score (0-1) based on:
     - Number of successful normalizations / total fields
     - Presence of identifiers (+0.2 score)
     - Author name resolution success rate
   - Log normalization failures to `normalization_errors.log` with field-level details
   
   **Optimizations**:
   - Parallelize normalization across references using multiprocessing (Python `joblib` or `concurrent.futures`)
   - Cache normalized venue mappings and author name expansions in Redis/Memcached
   - Use batch processing for embeddings (process 100+ titles at once)

2. **Fetching External Metadata**:
   
   **Purpose**: Retrieve authoritative reference metadata from external sources to validate extracted data.
   
   **API Source Prioritization** (Sequential fallback strategy):
   
   a. **Primary: Identifier-Based Retrieval** (highest accuracy)
      - **DOI via CrossRef**:
        - Endpoint: `https://api.crossref.org/works/{doi}`
        - Set polite pool headers: `User-Agent: YourApp/1.0 (mailto:your@email.com)`
        - Parse response: Extract title, authors (parse `given`/`family` names), published date, container-title (venue), ISSN, etc.
        - Response caching: 200 → cache for 30 days, 404 → cache for 24 hours, 5xx → retry with backoff
        - Timeout: 10 seconds
      - **arXiv ID via arXiv API**:
        - Endpoint: `http://export.arxiv.org/api/query?id_list={arxiv_id}`
        - Parse Atom XML: Extract title, authors, summary, categories, published/updated dates
        - Handle versioned IDs: Query without version, parse all versions from response
        - Batch queries: Fetch up to 100 IDs in single request (comma-separated)
      - **ISBN via OpenLibrary/Google Books**:
        - OpenLibrary: `https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data`
        - Google Books: `https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}`
        - Prefer OpenLibrary for open data, fallback to Google if not found
      - **PMID via PubMed E-utilities**:
        - Endpoint: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json`
        - Requires API key for >3 requests/second
   
   b. **Secondary: Fuzzy Search** (if no valid identifier or identifier lookup fails)
      - **Semantic Scholar**:
        - Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,year,venue,externalIds`
        - Query construction: `"{title}" {author_last_names} {year}` (use normalized title)
        - Limit: 10 results, sort by relevance
        - Filter: Year must match ±1 (e.g., if extracted year is 2017, accept 2016-2018)
        - API Key: Use S2 API key for higher rate limits (5000 req/5min)
      - **OpenAlex**:
        - Endpoint: `https://api.openalex.org/works?filter=title.search:{title}`
        - Add filters: `publication_year:{year},authorships.author.display_name:{author_names}`
        - Polite pool: Add email in query param `mailto`
        - Advantage: No API key needed, covers 250M+ works
      - **Google Scholar (Cautious Use)**:
        - Library: `scholarly` (unofficial scraper, use sparingly due to rate limits/blocking risk)
        - Query: `search_pubs(f"{title} {authors} {year}")`
        - Rate limiting: 1 request per 10 seconds, randomized delays
        - Use only as last resort due to instability
   
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
   
   d. **API Errors**:
      - **Rate Limiting (429)**:
        - Implement exponential backoff: Wait 2^n seconds (n = retry attempt, max 5 retries)
        - Switch to alternative API after max retries
        - Use distributed rate limiter (Redis-based) if parallel processing
      - **Timeout/Network Errors**:
        - Retry up to 3 times with 5-second timeout increments
        - Log error with traceback to `fetch_errors.log`
        - Mark as `fetch_status="error", error_type="timeout"`
      - **Invalid Response (e.g., malformed JSON)**:
        - Attempt to parse with lenient parser (e.g., `demjson`)
        - If fails, log raw response and skip
      - **Server Errors (5xx)**:
        - Retry after 60 seconds (once)
        - Fallback to next API in priority list
   
   **Caching Strategy** (Critical for performance):
   
   a. **Local Cache** (SQLite/PostgreSQL):
      - Schema: `CREATE TABLE metadata_cache (identifier TEXT PRIMARY KEY, api_source TEXT, metadata JSON, fetch_date TIMESTAMP, status TEXT)`
      - Key: Use identifier (DOI/arXiv) or hash of (title + authors + year) for fuzzy queries
      - TTL: 90 days for successful fetches, 7 days for failures (metadata may become available)
      - Invalidation: Purge cache for identifiers if user reports error
   
   b. **In-Memory Cache** (Redis):
      - For real-time processing, cache recent queries in Redis (TTL: 24 hours)
      - Key pattern: `metadata:{doi}` or `metadata:query:{hash}`
      - Benefit: <<1ms lookup vs. 200-500ms API call
   
   c. **Pre-fetching**:
      - For known datasets, pre-fetch metadata for common papers (e.g., highly-cited papers)
      - Build background worker to refresh cache for stale entries
   
   **Parallelization**:
   - Use async HTTP library (`aiohttp` or `httpx`) to fetch multiple references concurrently
   - Semaphore limit: max 10 concurrent requests per API (respect rate limits)
   - Batch processing: Process references in chunks of 100
   
   **Compliance & Ethics**:
   - Respect robots.txt and API terms of service
   - Include delay between requests (min 100ms for scholarly APIs)
   - Set descriptive User-Agent with contact email
   - Obtain API keys for production use (CrossRef Plus, Semantic Scholar)
   
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
   
   **Purpose**: Quantify similarity between extracted (E) and fetched (F) data for each field to identify discrepancies.
   
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
   
   - **Confidence Adjustment**:
     - If fetch_confidence < 0.8 (from fetching stage): Reduce overall_score by 10%
     - If identifiers match exactly: Boost overall_score by 5% (max 1.0)
   
   **Metric Storage** (for analysis):
   - Store all individual metric scores in JSON for each field:
     ```json
     {
       "title_metrics": {"levenshtein": 0.92, "jaro": 0.94, "semantic": 0.88, "ensemble": 0.91},
       "author_metrics": {"jaccard": 0.85, "sequence": 0.90, "ensemble": 0.87},
       ...
     }
     ```
   - Enables offline analysis, threshold tuning, and A/B testing
   
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
   
   **Purpose**: Translate continuous similarity scores into discrete, actionable verdicts with confidence assessments.
   
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
      1. **Identifier match override**: If DOI or arXiv exact match → minimum verdict = "likely_verified" (even if other fields mismatch slightly)
      2. **Hard mismatch veto**: Any hard_mismatch in title OR authors → maximum verdict = "questionable"
      3. **Year critical**: Year hard_mismatch + no identifier → maximum verdict = "unverified" (likely fabrication)
      4. **Fetch failure**: If fetch_status = "error" → verdict = "unfetchable" regardless of score
   
   d. **Quality Flags and Warnings** (Metadata for downstream use):
      - Generate actionable warnings:
        - `"low_confidence"`: avg_confidence < 0.70
        - `"incomplete_data"`: >2 fields have "incomplete" verdict
        - `"fetch_ambiguous"`: Multiple external matches with similar scores
        - `"normalization_issues"`: Normalization confidence < 0.60
        - `"identifier_mismatch"`: Identifiers present but don't match (critical red flag)
        - `"year_outlier"`: Year difference >5 years
        - `"author_count_mismatch"`: Significant difference in author count (e.g., 1 author vs. 10)
      - Severity levels: `"info"`, `"warning"`, `"error"`
   
   e. **Dispute Resolution Logic** (When metrics/rules conflict):
      - If identifier matches but other fields mismatch:
        - Trust identifier → verdict ≥ "likely_verified"
        - Add warning: `"field_identifier_conflict"`
      - If title/authors match but venue differs significantly:
        - Possible preprint vs. published version → check year proximity
        - If year similar, verdict = "partially_verified" with flag `"venue_update_suspected"`
      - If year differs but all else matches:
        - Check if one is preprint (arXiv) and other published → acceptable
        - Add flag: `"version_difference_detected"`
   
   f. **Explainability** (Human-readable verdict explanation):
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
   
   **Output Structure**:
   ```json
   {
     "overall_score": 0.87,
     "overall_verdict": "likely_verified",
     "overall_confidence": 0.82,
     "verdict_explanation": "Reference likely verified with 87% similarity...",
     "quality_flags": [
       {"level": "warning", "code": "author_et_al_truncation", "message": "Author list truncated with et al."}
     ],
     "verdict_metadata": {
       "timestamp": "2026-02-12T10:30:00Z",
       "version": "1.0",
       "weights_used": {"title": 0.35, "authors": 0.30, ...},
       "thresholds_used": {"title": 0.95, "authors": 0.90, ...}
     }
   }
   ```

5. **Output Formatting**:
   
   **Purpose**: Provide comprehensive, structured output suitable for downstream applications, debugging, and human review.
   
   **Output Schema Design Principles**:
   - **Completeness**: Include all intermediate data (extracted, fetched, normalized, scores)
   - **Traceability**: Maintain provenance (timestamps, sources, versions)
   - **Actionability**: Provide clear verdicts and recommended actions
   - **Debuggability**: Store metric breakdowns and error logs
   - **Extensibility**: Schema versioning for backward compatibility
   
   **Complete JSON Schema** (per reference):
   
   ```json
   {
     "pipeline_metadata": {
       "version": "1.0.0",
       "execution_timestamp": "2026-02-12T10:30:00Z",
       "processing_time_ms": 324,
       "paper_id": "input_paper_123",
       "paper_metadata": {
         "title": "Paper title if available",
         "doi": "10.xxxx/xxxxx"
       }
     },
     "references": [
       {
         "id": "1",
         "reference_index": 1,  // Position in original paper
         
         // === EXTRACTION STAGE ===
         "raw_text": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).",
         "extraction_metadata": {
           "method": "grobid",
           "grobid_version": "0.7.3",
           "extraction_confidence": 0.92,
           "extraction_timestamp": "2026-02-12T10:29:45Z",
           "extraction_errors": []  // Any parsing warnings
         },
         "parsed_data": {
           "title": "Attention is all you need",
           "title_original": "Attention is all you need",  // Before normalization
           "authors": [
             {"first": "Ashish", "middle": null, "last": "Vaswani", "full": "Ashish Vaswani"},
             {"first": "Noam", "middle": null, "last": "Shazeer", "full": "Noam Shazeer"},
             {"first": "Niki", "middle": null, "last": "Parmar", "full": "Niki Parmar"},
             {"first": "Jakob", "middle": null, "last": "Uszkoreit", "full": "Jakob Uszkoreit"},
             "..."  // Truncated for brevity
           ],
           "authors_truncated": false,
           "year": 2017,
           "venue": {
             "raw": "Advances in neural information processing systems",
             "normalized": "Neural Information Processing Systems",
             "abbreviation": "NeurIPS",
             "type": "conference"
           },
           "pages": "5998-6008",
           "volume": null,
           "issue": null,
           "identifiers": {
             "doi": null,
             "arxiv_id": null,  // Not found in raw text
             "isbn": null,
             "pmid": null
           }
         },
         
         // === NORMALIZATION STAGE ===
         "normalized_data": {
           "title_normalized": "attention all need",  // Lowercased, stemmed
           "title_embedding": "[0.123, -0.456, ...]",  // SciBERT embedding (truncated)
           "authors_normalized": [
             {"first": "Ashish", "last": "Vaswani", "last_normalized": "vaswani"},
             "..."
           ],
           "year_normalized": 2017,
           "venue_normalized": "neurips",
           "normalization_confidence": 0.88,
           "normalization_warnings": [
             {"field": "identifiers", "message": "No DOI found in raw text"}
           ]
         },
         
         // === FETCHING STAGE ===
         "fetched_data": {
           "source_api": "semantic_scholar",
           "fetch_status": "single_match",
           "fetch_confidence": 0.95,
           "fetch_timestamp": "2026-02-12T10:29:58Z",
           "fetch_duration_ms": 234,
           "query_used": '"attention all need" vaswani shazeer 2017',
           "num_candidates": 1,
           
           // Retrieved metadata
           "title": "Attention is All You Need",
           "authors": [
             {"first": "Ashish", "last": "Vaswani"},
             {"first": "Noam", "last": "Shazeer"},
             {"first": "Niki", "last": "Parmar"},
             {"first": "Jakob", "last": "Uszkoreit"},
             {"first": "Llion", "last": "Jones"},
             {"first": "Aidan N.", "last": "Gomez"},
             {"first": "Łukasz", "last": "Kaiser"},
             {"first": "Illia", "last": "Polosukhin"}
           ],
           "year": 2017,
           "venue": {
             "name": "Neural Information Processing Systems",
             "abbreviation": "NeurIPS",
             "type": "conference",
             "year": 2017
           },
           "identifiers": {
             "doi": "10.5555/3295222.3295349",
             "arxiv_id": "1706.03762",
             "semantic_scholar_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
             "corpus_id": 13756489
           },
           "citation_count": 75432,
           "influential_citation_count": 8765,
           "url": "https://arxiv.org/abs/1706.03762",
           
           // Alternative matches (if fetch_status = "ambiguous_match")
           "alternative_matches": [
             // {"title": "...", "score": 0.72, ...}
           ]
         },
         
         // === SIMILARITY COMPUTATION ===
         "similarity_scores": {
           "title": {
             "overall_score": 0.96,
             "metrics": {
               "levenshtein": 0.95,
               "jaro_winkler": 0.97,
               "ratcliff_obershelp": 0.96,
               "jaccard": 0.94,
               "tfidf_cosine": 0.95,
               "semantic_cosine": 0.98
             },
             "metric_variance": 0.012,  // Low variance = high agreement
             "ensemble_method": "weighted_average"
           },
           "authors": {
             "overall_score": 0.85,
             "metrics": {
               "last_name_jaccard": 0.80,  // Some authors missing in extraction
               "first_initial_match": 0.88,
               "sequence_similarity": 0.90,
               "full_name_match": 0.83
             },
             "matched_authors": 6,
             "total_extracted": 6,
             "total_fetched": 8,
             "missing_in_extracted": ["Llion Jones", "Illia Polosukhin"]
           },
           "year": {
             "overall_score": 1.0,
             "exact_match": true,
             "difference": 0
           },
           "venue": {
             "overall_score": 0.98,
             "metrics": {
               "exact_match": false,
               "abbreviation_match": true,
               "fuzzy_match": 0.92,
               "semantic_match": 0.95
             },
             "match_type": "abbreviation"
           },
           "identifiers": {
             "overall_score": 0.5,
             "doi_match": null,  // N/A (not present in both)
             "arxiv_match": null,
             "completeness": 0.5,  // Only fetched has identifiers
             "note": "Identifiers missing in extracted data"
           }
         },
         
         // === VERDICT ASSIGNMENT ===
         "field_verdicts": {
           "title": {
             "score": 0.96,
             "verdict": "match",
             "confidence": 0.91,
             "threshold_used": 0.95,
             "flags": ["case_difference_only"],
             "explanation": "Titles match with only capitalization differences."
           },
           "authors": {
             "score": 0.85,
             "verdict": "soft_mismatch",
             "confidence": 0.78,
             "threshold_used": 0.90,
             "flags": ["authors_missing_in_extraction"],
             "explanation": "6 of 8 authors matched. Missing: Llion Jones, Illia Polosukhin."
           },
           "year": {
             "score": 1.0,
             "verdict": "match",
             "confidence": 1.0,
             "threshold_used": 1.0,
             "flags": [],
             "explanation": "Years match exactly (2017)."
           },
           "venue": {
             "score": 0.98,
             "verdict": "match",
             "confidence": 0.95,
             "threshold_used": 0.95,
             "flags": ["abbreviation_resolved"],
             "explanation": "Venue matched via abbreviation expansion (NeurIPS)."
           },
           "identifiers": {
             "score": 0.5,
             "verdict": "incomplete",
             "confidence": 0.60,
             "threshold_used": 1.0,
             "flags": ["identifiers_not_extracted"],
             "explanation": "Identifiers found in external source but missing in extraction."
           }
         },
         
         "overall_assessment": {
           "overall_score": 0.89,
           "overall_verdict": "likely_verified",
           "overall_confidence": 0.85,
           "verdict_explanation": "Reference likely verified with 89% similarity. Minor author list incompleteness detected. Title, year, and venue match well. Identifiers could not be extracted but were found externally.",
           
           "quality_flags": [
             {"level": "info", "code": "author_truncation", "message": "2 authors missing from extraction"},
             {"level": "warning", "code": "identifier_incomplete", "message": "Identifiers not extracted from reference text"}
           ],
           
           "recommended_action": "accept",  // One of: ["accept", "review", "reject"]
           "confidence_tier": "high",  // One of: ["very_high", "high", "medium", "low", "very_low"]
           
           "weights_used": {
             "title": 0.35,
             "authors": 0.30,
             "year": 0.15,
             "venue": 0.10,
             "identifiers": 0.10
           },
           
           "adjustment_factors": [
             {"type": "fetch_confidence_penalty", "impact": -0.02},
             {"type": "identifier_boost", "impact": 0.0}  // No boost (not matched)
           ]
         },
         
         // === DEBUGGING & METADATA ===
         "processing_logs": [
           {"stage": "extraction", "timestamp": "2026-02-12T10:29:45Z", "status": "success", "duration_ms": 120},
           {"stage": "normalization", "timestamp": "2026-02-12T10:29:50Z", "status": "success", "duration_ms": 45},
           {"stage": "fetching", "timestamp": "2026-02-12T10:29:58Z", "status": "success", "duration_ms": 234},
           {"stage": "similarity", "timestamp": "2026-02-12T10:30:12Z", "status": "success", "duration_ms": 78},
           {"stage": "verdict", "timestamp": "2026-02-12T10:30:18Z", "status": "success", "duration_ms": 23}
         ],
         
         "errors": [],  // Any errors encountered during processing
         "warnings": [
           {"code": "W001", "message": "DOI not found in raw reference text", "stage": "extraction"},
           {"code": "W002", "message": "Author count mismatch (6 vs 8)", "stage": "similarity"}
         ]
       }
       // Additional references...
     ],
     
     // === AGGREGATE STATISTICS ===
     "summary_statistics": {
       "total_references": 45,
       "verified": 32,
       "likely_verified": 8,
       "partially_verified": 3,
       "questionable": 1,
       "unverified": 0,
       "unfetchable": 1,
       "average_overall_score": 0.87,
       "average_confidence": 0.83,
       "processing_time_total_ms": 14580,
       "processing_time_per_reference_ms": 324
     },
     
     // === BENCHMARKING DATA (if ground truth available) ===
     "benchmark_results": {
       "accuracy": 0.91,
       "precision": 0.89,
       "recall": 0.94,
       "f1_score": 0.91,
       "field_level_metrics": {
         "title": {"precision": 0.95, "recall": 0.97, "f1": 0.96},
         "authors": {"precision": 0.88, "recall": 0.91, "f1": 0.89},
         "year": {"precision": 0.99, "recall": 0.99, "f1": 0.99},
         "venue": {"precision": 0.86, "recall": 0.89, "f1": 0.87},
         "identifiers": {"precision": 0.92, "recall": 0.85, "f1": 0.88}
       }
     }
   }
   ```
   
   **Output Formats** (Multiple export options):
   
   1. **JSON** (primary, machine-readable):
      - Full schema as shown above
      - Compressed variant: Omit metric breakdowns, keep only verdicts
   
   2. **CSV** (tabular, for spreadsheet analysis):
      - Flattened structure: One row per reference
      - Columns: reference_id, raw_text, extracted_title, fetched_title, title_score, title_verdict, ..., overall_score, overall_verdict
   
   3. **HTML Report** (human-readable):
      - Visual dashboard with color-coded verdicts
      - Side-by-side comparison of extracted vs. fetched data
      - Highlights for mismatches
   
   4. **Markdown Summary**:
      - Text report for quick review
      - Lists problematic references (questionable/unverified)
   
   **Version Control & Schema Evolution**:
   - Include schema version in output: `"schema_version": "1.0.0"`
   - Maintain backward compatibility for minor versions
   - Document breaking changes in major versions

### Verification Benchmark

**Purpose**: Rigorously evaluate verification system accuracy, reliability, and robustness using standardized protocols and diverse test datasets.

#### 1. **Dataset Construction**

**Ground Truth Sources** (Multi-source validation):
- **Gold Standard**: 1,000+ references from curated sources:
  - DBLP (computer science): High-quality, manually curated metadata
  - PubMed (biomedical): Trusted medical literature
  - arXiv (preprints): Version tracking, author-provided metadata
  - ACL Anthology (NLP/CL): Conference proceedings
- **Manual Annotation**: 300+ references manually validated by domain experts
  - Annotate extraction errors, ambiguous cases, edge cases
  - Inter-annotator agreement: Krippendorff's α > 0.85 required

**Dataset Composition** (Stratified sampling):

a. **By Venue Type** (balanced representation):
   - Conferences: 40%
   - Journals: 35%
   - Preprints (arXiv, bioRxiv): 15%
   - Books/Theses: 5%
   - Other (tech reports, websites): 5%

b. **By Quality Level** (test robustness):
   - Clean references (perfect extraction): 60%
   - OCR errors (scanned papers): 15%
   - Formatting issues (non-standard styles): 15%
   - Truncated/incomplete (missing fields): 10%

c. **By Era** (temporal coverage):
   - Recent (2020-2026): 40%
   - Modern (2010-2019): 35%
   - Historical (2000-2009): 15%
   - Legacy (pre-2000): 10%

d. **Edge Cases** (stress testing) - 200+ references:
   - Fabricated references (non-existent papers): 50
   - Author name variants (married names, transliterations): 40
   - Venue name changes (NIPS→NeurIPS): 30
   - Preprint vs. published versions: 40
   - Retracted papers: 20
   - Self-published/grey literature: 20

**Data Split**:
- **Development Set**: 60% (for threshold tuning, parameter optimization)
- **Validation Set**: 20% (for hyperparameter selection, early stopping)
- **Test Set**: 20% (for final evaluation, never seen during development)
- Ensure no overlap: Same paper cannot appear in multiple splits

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
  },
  "metadata": {
    "difficulty": "medium",  // easy, medium, hard
    "category": "ocr_error",
    "annotator_id": "expert_01",
    "annotation_date": "2026-01-15"
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

c. **Confusion Matrix Analysis**:
   - Build 5×5 matrix for each field (actual vs. predicted)
   - Identify systematic errors:
     - "match" predicted as "soft_mismatch": Threshold too strict
     - "hard_mismatch" predicted as "match": Critical error (high risk)
   - Compute error rates:
     - False positive rate: FP / (FP + TN)
     - False negative rate: FN / (TP + FN)

d. **Field-Specific Metrics**:
   - **Title**:
     - Exact match rate: % with Levenshtein = 1.0
     - Semantic clustering: % that cluster correctly with ground truth (t-SNE visualization)
   - **Authors**:
     - Author-level precision/recall (treat each author as unit)
     - First-author accuracy: % where first author correctly identified
   - **Venue**:
     - Abbreviation resolution rate: % correctly expanded
     - Type classification accuracy: Conference vs. journal distinction
   - **Identifiers**:
     - Extraction success rate: % where DOI/arXiv extracted from raw text
     - Validation success rate: % where extracted identifier confirmed via API

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

d. **Confidence Calibration**:
   - **Calibration Plot**: Predicted confidence vs. actual accuracy
     - Bin predictions by confidence (0-0.1, 0.1-0.2, ..., 0.9-1.0)
     - Plot: Mean predicted confidence vs. empirical accuracy per bin
     - Well-calibrated: Points lie on diagonal (y=x)
   - **Expected Calibration Error (ECE)**:
     - Weighted average of accuracy-confidence gaps across bins
     - ECE < 0.05: Well-calibrated
   - **Brier Score**: Mean squared error of confidence predictions

**Error Analysis Framework**:

a. **Error Taxonomy** (Categorize all errors):
   - **Extraction errors**: GROBID parsing failures, OCR issues
   - **Normalization errors**: Author name resolution, venue abbreviation
   - **Fetching errors**: API timeout, wrong match selected
   - **Similarity errors**: Thresholds too strict/loose, embedding failure
   - **Verdict errors**: Incorrect logic, weight misconfiguration

b. **Per-Error Metrics**:
   - **Error rate by type**: % of errors in each category
   - **Severity distribution**: How many critical (hard_mismatch) vs. minor (soft_mismatch)?
   - **Downstream impact**: How many errors lead to wrong overall verdict?

c. **Qualitative Analysis** (Manual review):
   - Sample 50 errors from each category
   - Expert review: Is error system fault or ground truth ambiguity?
   - Document failure modes and propose fixes

#### 3. **Benchmarking Protocols**

**Baseline Comparisons**:

a. **Naive Baselines**:
   - **String Matching**: Exact string match on title/authors
   - **Levenshtein-Only**: Single metric (no ensemble)
   - **Rule-Based**: Simple if-else thresholds (no ML)

b. **External Systems** (if available):
   - **GROBID's Built-in Validation**: Use GROBID confidence scores
   - **anystyle.io**: Reference parsing and matching
   - **Crossref Similarity Search**: Direct API similarity

c. **Ablation Studies** (Measure component contribution):
   - **Without normalization**: Raw extracted data vs. fetched
   - **Without embeddings**: Only string metrics (no semantic)
   - **Without ensemble**: Single best metric per field
   - **Without caching**: Measure performance impact
   - **Without author disambiguation**: No name resolution
   - Report: Δ F1 for each ablation (e.g., "-8% F1 without normalization")

**Cross-Validation**:
- **k-Fold CV** (k=5): Partition dataset into 5 folds
- Train/tune on 4 folds, test on 1; rotate
- Report: Mean ± std of metrics across folds
- Benefit: Robust estimates, detect overfitting

**Statistical Significance Testing**:
- **Paired t-test**: Compare proposed vs. baseline on same test set
- **McNemar's test**: For binary outcomes (correct/incorrect)
- **Bootstrap resampling**: Compute 95% confidence intervals
- Report: p-value, effect size (Cohen's d)
- Significance threshold: p < 0.05

**Robustness Testing**:

a. **Adversarial Cases**:
   - **Fabricated references**: Ensure system detects non-existent papers
     - Target: >95% unverified verdict for fabrications
   - **Near-duplicates**: Slightly altered title/authors (typosquatting)
     - Ensure system doesn't false-positive
   - **Ambiguous queries**: Generic titles (e.g., "Introduction")
     - System should flag as ambiguous

b. **Out-of-Distribution Testing**:
   - Test on different domains (e.g., train on CS, test on biology)
   - Test on different time periods (train on 2010s, test on 2020s)
   - Test on different formats (train on ACM, test on IEEE)
   - Report: Performance degradation (Δ F1)

c. **Noise Injection**:
   - Add synthetic OCR errors (character substitution, deletion)
   - Levels: 5%, 10%, 20% character error rate
   - Measure degradation: Plot F1 vs. noise level

**Efficiency Benchmarks**:
- **Throughput**: References processed per second
- **Latency**: Time per reference (p50, p95, p99 percentiles)
- **API call efficiency**: Cache hit rate, API requests per reference
- **Resource usage**: CPU, memory, network bandwidth
- Compare: With vs. without caching, with vs. without parallelization

#### 4. **Reporting Standards**

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

3. **Confusion Matrices**: Visual heatmaps for each field

4. **Threshold Configuration**:
   - Document all thresholds used (per field, overall)
   - Justify choices (e.g., "Selected 0.95 for title to maximize precision")

5. **Ablation Results**:
   ```
   | Ablation                  | F1 Change | Conclusion               |
   |---------------------------|-----------|-------------------------|
   | Without normalization     | -0.08     | Critical component      |
   | Without embeddings        | -0.05     | Important for semantics |
   | Without ensemble          | -0.03     | Modest improvement      |
   | Single API (no fallback)  | -0.06     | Fallback necessary      |
   ```

6. **Error Analysis Summary**:
   - Top-5 error types with frequencies
   - Examples of each error type
   - Proposed mitigation strategies

7. **Comparison to Baselines**:
   - Table comparing proposed vs. baselines
   - Statistical significance indicators (*, **, ***)

8. **Calibration Plots**: Confidence vs. accuracy visualization

9. **Edge Case Performance**:
   - Fabricated: X% detected
   - OCR errors: Y F1 degradation
   - Historical papers: Z F1

10. **Computational Cost**:
    - Average time per reference: Xms
    - Throughput: Y refs/sec
    - API costs: Z calls per reference

**Publication-Ready Benchmark**:
- Follow SIGIR/ACL/ICML paper standards
- Include reproducibility checklist:
  - Code availability (GitHub link)
  - Dataset availability (Zenodo/Figshare)
  - Hyperparameters and random seeds
  - Hardware specs (CPU/GPU, RAM)
- Provide supplementary materials:
  - Full confusion matrices
  - Per-reference predictions (for error analysis)
  - Calibration data

**Continuous Benchmarking**:
- Version control: Track metrics across system versions
- Regression testing: Ensure new features don't degrade performance
- Dashboard: Real-time monitoring of benchmark metrics
- Quarterly re-evaluation: As APIs/datasets evolve