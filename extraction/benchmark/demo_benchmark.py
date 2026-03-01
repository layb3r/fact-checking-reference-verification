#!/usr/bin/env python3
"""
Demo Script - Shows how to use the benchmarking system
"""

from extraction_benchmark.benchmark_extraction import ExtractionBenchmark
import json

def demo_simple():
    """Simple demonstration with a few references"""
    print("=" * 80)
    print("DEMO: Simple Benchmarking Example")
    print("=" * 80)
    
    # Ground truth (what should be extracted)
    ground_truth = [
        {
            "title": "Attention is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": "2017",
            "venue": "NeurIPS",
        },
        {
            "title": "Deep Learning",
            "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
            "year": "2016",
            "venue": "MIT Press",
        }
    ]
    
    # Extracted data (with some variations and errors)
    extracted = [
        {
            "title": "Attention Is All You Need",  # Case difference
            "authors": ["A. Vaswani", "N. Shazeer", "N. Parmar"],  # Abbreviated names
            "year": "2017",
            "venue": "Neural Information Processing Systems",  # Full name vs abbreviation
        },
        {
            "title": "Deep Learning Book",  # Title variation
            "authors": ["I. Goodfellow", "Y. Bengio"],  # Missing one author
            "year": "2016",
            "venue": "MIT Press",
        }
    ]
    
    fields = ["title", "authors", "year", "venue"]
    
    # Test with different methods
    print("\n1. STRICT Matching (exact match only)")
    print("-" * 80)
    benchmark_strict = ExtractionBenchmark(matching_method="strict")
    benchmark_strict.compare_references(extracted, ground_truth, fields)
    print(benchmark_strict.generate_report())
    
    print("\n2. SOFT Matching (normalized, case-insensitive)")
    print("-" * 80)
    benchmark_soft = ExtractionBenchmark(matching_method="soft")
    benchmark_soft.compare_references(extracted, ground_truth, fields)
    print(benchmark_soft.generate_report())
    
    print("\n3. LEVENSHTEIN Matching (tolerates typos)")
    print("-" * 80)
    benchmark_lev = ExtractionBenchmark(matching_method="levenshtein")
    benchmark_lev.compare_references(extracted, ground_truth, fields)
    print(benchmark_lev.generate_report())
    
    # Save results
    benchmark_soft.save_results("demo_results.json")
    print("\n✓ Results saved to demo_results.json")


def demo_comparison():
    """Show method comparison"""
    print("\n" + "=" * 80)
    print("DEMO: Comparing Different Matching Methods")
    print("=" * 80)
    
    # Some test data
    ground_truth = [
        {"title": "Machine Learning", "year": "2020"},
        {"title": "Neural Networks", "year": "2019"},
        {"title": "Deep Learning", "year": "2021"},
    ]
    
    extracted = [
        {"title": "Machine Learning", "year": "2020"},      # Perfect match
        {"title": "neural networks", "year": "2019"},      # Case different
        {"title": "Deep Learning Methods", "year": "2021"}, # Partial match
    ]
    
    methods = ["strict", "soft", "levenshtein"]
    results_table = []
    
    for method in methods:
        benchmark = ExtractionBenchmark(matching_method=method)
        results = benchmark.compare_references(extracted, ground_truth, ["title", "year"])
        
        results_table.append({
            "method": method,
            "title_f1": results["title"].f1,
            "year_f1": results["year"].f1,
        })
    
    # Print comparison
    print("\nF1-Score Comparison:")
    print(f"{'Method':<15} {'Title F1':<12} {'Year F1':<12}")
    print("-" * 40)
    for row in results_table:
        print(f"{row['method']:<15} {row['title_f1']:<12.2f} {row['year_f1']:<12.2f}")
    
    print("\nKey Insights:")
    print("  • Strict: Only matches 1/3 titles (exact match)")
    print("  • Soft: Matches 2/3 titles (case-insensitive)")
    print("  • Levenshtein: Matches 2-3/3 titles (tolerates variations)")
    print("  • Year: All methods match perfectly (exact values)")


def demo_field_level_analysis():
    """Show detailed field-level analysis"""
    print("\n" + "=" * 80)
    print("DEMO: Field-Level Analysis")
    print("=" * 80)
    
    # Realistic scenario with mixed quality
    ground_truth = [
        {
            "title": "Attention is All You Need",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "year": "2017",
            "doi": "10.5555/3295222.3295349"
        }
    ]
    
    extracted = [
        {
            "title": "Attention is all you need",  # Good (case diff)
            "authors": ["Vaswani", "Shazeer"],     # Missing one author
            "year": "2017",                         # Perfect
            "doi": "10.5555/3295222.3295349"       # Perfect
        }
    ]
    
    benchmark = ExtractionBenchmark(matching_method="soft")
    results = benchmark.compare_references(extracted, ground_truth, 
                                          ["title", "authors", "year", "doi"])
    
    print("\nPer-Field Analysis:")
    print("-" * 80)
    for field, result in results.items():
        status = "✓ GOOD" if result.f1 >= 0.90 else "⚠ ISSUE" if result.f1 >= 0.70 else "✗ BAD"
        print(f"{field:<12} F1={result.f1:.2f}  P={result.precision:.2f}  R={result.recall:.2f}  {status}")
        
        if result.f1 < 0.90:
            if result.missing > 0:
                print(f"             → {result.missing} value(s) not extracted")
            if result.mismatches > 0:
                print(f"             → {result.mismatches} mismatch(es)")


if __name__ == "__main__":
    # Run all demos
    demo_simple()
    demo_comparison()
    demo_field_level_analysis()
    
    print("\n" + "=" * 80)
    print("To use with your own data:")
    print("  python run_benchmark.py -e your_extracted.json -g your_ground_truth.json")
    print("=" * 80)
