from claimcheck import ReferenceChecker
import json


def test_simple_citation():
    """Test the ReferenceChecker with a simple example citation."""
    
    # Load environment variables
    dotenv.load_dotenv("../.env")
    
    # Initialize checker
    checker = ReferenceChecker(llm_provider="openai", embedding_provider="local")
    
    # Example citation and reference
    citation = "The study found a 25% increase in performance after the intervention."
    reference_text = """
    Methods:
    The intervention was administered to 100 participants over 6 weeks.
    
    Results:
    Analysis showed that participants demonstrated a 25.3% improvement in 
    performance metrics (p < 0.001) following the 6-week intervention period.
    
    Discussion:
    The observed 25% increase in performance represents a significant improvement
    and aligns with previous studies in this domain.
    """
    
    result = checker.check_citation(citation, reference_text)
    print(json.dumps(result, indent=2))
    
    # Verify result has expected structure
    assert "citation_text" in result
    assert "classification" in result
    assert "reasoning" in result
    assert "evidence" in result
    assert "metadata" in result
    
    # Verify classification is valid
    assert result['classification'] in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]
