interface CitationMetadata {
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

interface BenchmarkGroundTruth {
    true_existence: 0 | 1;
    true_hallucination_category: HallucinationTaxonomy | null;
    true_alignment: 0 | 1 | 2 | null;
    expert_rationale: string;
}

// - 0: supported (fully aligned): Citation claim is fully supported by the reference
// - 1: unsupported (misaligned): Citation claim contradicts or is not supported by the reference
// - 2: uncertain (ambiguous alignment): Insufficient information to determine support level

interface DatasetInstance {
    claim_text: string;
    surrounding_context: string;
    citation_metadata: CitationMetadata;
    true_outputs: BenchmarkGroundTruth;
}

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

/*

The collection process employed a stratified sampling approach across eight academic fields (Computer
Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, and Psychology)

*/