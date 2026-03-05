/**
 * Root Object: Represents a single claim-citation verification instance.
 */
interface VerificationInstance {
  instance_id: string;             // UUIDv4
  timestamp: string;               // ISO 8601 UTC
  status: AgentExecutionState;     // Current status in the multi-agent pipeline
  
  // 1. INPUT: The raw context extracted from the source document
  document_context: DocumentContext;
  
  // 2. EXTRACTION: The structured metadata parsed by the Extractor Agent
  extracted_citation: ExtractedCitation;
  
  // 3. RETRIEVAL: The evidence gathered by Memory, Web, and Scholar Agents
  retrieval_traces: RetrievalTrace[];
  
  // 4. ADJUDICATION: The final verdict by the Judge Agent
  system_prediction: SystemPrediction;
  
  // 5. BENCHMARKING (Optional): The gold-standard labels for evaluation
  ground_truth?: BenchmarkGroundTruth;
}

/**
 * Tracks the real-time state of the verification pipeline.
 */
enum AgentExecutionState {
  PENDING = "PENDING",
  EXTRACTING = "EXTRACTING",
  RETRIEVING_MEMORY = "RETRIEVING_MEMORY",
  RETRIEVING_WEB = "RETRIEVING_WEB",
  RETRIEVING_SCHOLAR = "RETRIEVING_SCHOLAR",
  JUDGING = "JUDGING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED"
}

/**
 * Represents the claim and its surrounding text within the PDF.
 */
interface DocumentContext {
  source_document_id: string;
  raw_citation_string: string;
  claim_text: string;
  surrounding_context: string;
  visual_coordinates?: {           // Bounding boxes for multimodal UI highlighting
    page: number;
    bbox: [number, number, number, number]; 
  }[];
}

/**
 * Structured metadata M_i mapped from the raw citation.
 */
interface ExtractedCitation {
  title: string | null;
  authors: string[];
  venue: string | null;
  year: number | null;
  identifiers: {
    doi: string | null;
    arxiv_id: string | null;
    url: string | null;
  };
}

/**
 * Logs the evidence retrieved by external tools to ensure auditability.
 */
interface RetrievalTrace {
  agent_type: "Memory" | "Web" | "Scholar";
  execution_time_ms: number;
  retrieved_metadata: ExtractedCitation | null; // The canonical record found
  full_text_scraped: boolean;                   // Indicates if deep-crawling was successful
  evidence_snippets: string[];                  // Passages specifically related to the claim
  source_uri: string;
  confidence_score?: number;                    // Similarity score (e.g., from vector search)
}

/**
 * The final output based on the Strict Consistency Criterion and Alignment Logic.
 */
interface SystemPrediction {
  existence: {
    label: 0 | 1;                  // 1 = Exists & Accurate, 0 = Hallucinated/Incorrect
    hallucination_category: HallucinationTaxonomy | null; // Required if label === 0
    strict_match_flags: {          // Component-level boolean checks
      title_match: boolean;
      author_match: boolean;
      venue_match: boolean;
      year_match: boolean;
    };
  };
  alignment: {
    label: 0 | 1 | 2 | null;   // 0: Supported, 1: Unsupported, 3: Uncertain. Null if existence === 0
    confidence_score: number;      // 0.0 to 1.0 calibration score
  };
  explanation: {
    reasoning_trace: string;       // Natural language justification
    cited_evidence_indices: number[]; // Pointers to specific evidence_snippets
  };
}

/**
 * Categorization of fabricated or erroneous citations.
 */
enum HallucinationTaxonomy {
  TITLE_ERROR_SUBSTITUTE = "TITLE_ERROR_SUBSTITUTE",
  TITLE_ERROR_PARAPHRASE = "TITLE_ERROR_PARAPHRASE",
  TITLE_ERROR_FULLY_FABRICATED = "TITLE_ERROR_FULLY_FABRICATED",
  AUTHOR_ERROR_ADD_DEL = "AUTHOR_ERROR_ADD_DEL",
  AUTHOR_ERROR_PERTURBATION = "AUTHOR_ERROR_PERTURBATION",
  AUTHOR_ERROR_FULLY_FABRICATED = "AUTHOR_ERROR_FULLY_FABRICATED",
  META_ERROR_DOI = "META_ERROR_DOI",
  META_ERROR_DATE = "META_ERROR_DATE",
  META_ERROR_VENUE = "META_ERROR_VENUE",
  COMPOUND_ERROR = "COMPOUND_ERROR"
}

/**
 * Human-annotated labels for benchmark evaluation and metric computation.
 */
interface BenchmarkGroundTruth {
  true_existence: 0 | 1;
  true_hallucination_category: HallucinationTaxonomy | null;
  true_alignment: 0 | 1 | 2 | null;
  expert_rationale: string;
}

