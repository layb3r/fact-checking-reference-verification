#!/usr/bin/env python3
"""
Benchmark Runner
Loads extracted and ground truth data, runs benchmarks, generates reports.
"""

import json
import argparse
from pathlib import Path
from benchmark_extraction import ExtractionBenchmark

def load_json_file(file_path: str) -> list:
    """Load JSON file and return list of references"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict) and 'references' in data:
        return data['references']
    elif isinstance(data, list):
        return data
    else:
        return [data]


def run_benchmark_on_files(extracted_path: str, ground_truth_path: str, 
                           output_dir: str = "benchmark_results"):
    """
    Run benchmark comparing extracted and ground truth files.
    
    Args:
        extracted_path: Path to extracted references JSON
        ground_truth_path: Path to ground truth JSON
        output_dir: Directory to save results
    """
    # Load data
    print(f"Loading extracted data from: {extracted_path}")
    extracted_refs = load_json_file(extracted_path)
    
    print(f"Loading ground truth from: {ground_truth_path}")
    ground_truth_refs = load_json_file(ground_truth_path)
    
    print(f"\nLoaded {len(extracted_refs)} extracted references")
    print(f"Loaded {len(ground_truth_refs)} ground truth references")
    
    # Determine fields to compare (intersection of available fields)
    if extracted_refs and ground_truth_refs:
        extracted_fields = set(extracted_refs[0].keys()) - {'raw_text', 'id'}
        gt_fields = set(ground_truth_refs[0].keys()) - {'raw_text', 'id', 'reference_id', 'metadata'}
        common_fields = sorted(extracted_fields & gt_fields)
        
        print(f"\nComparing fields: {', '.join(common_fields)}")
    else:
        print("Error: No references to compare")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run benchmark with different matching methods
    methods = ["strict", "soft", "levenshtein", "ratcliff"]
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running benchmark with {method.upper()} matching")
        print(f"{'='*80}")
        
        benchmark = ExtractionBenchmark(matching_method=method)
        benchmark.compare_references(extracted_refs, ground_truth_refs, common_fields)
        
        # Print report
        report = benchmark.generate_report()
        print(report)
        
        # Save results
        result_file = output_path / f"results_{method}.json"
        benchmark.save_results(str(result_file))
        print(f"\nResults saved to: {result_file}")
        
        # Save text report
        report_file = output_path / f"report_{method}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")


def compare_matching_methods(extracted_path: str, ground_truth_path: str):
    """Generate comparison table across all matching methods"""
    
    extracted_refs = load_json_file(extracted_path)
    ground_truth_refs = load_json_file(ground_truth_path)
    
    # Get common fields
    if extracted_refs and ground_truth_refs:
        extracted_fields = set(extracted_refs[0].keys()) - {'raw_text', 'id'}
        gt_fields = set(ground_truth_refs[0].keys()) - {'raw_text', 'id', 'reference_id', 'metadata'}
        common_fields = sorted(extracted_fields & gt_fields)
    else:
        return
    
    methods = ["strict", "soft", "levenshtein", "ratcliff"]
    comparison = {field: {} for field in common_fields}
    
    # Run all methods
    for method in methods:
        benchmark = ExtractionBenchmark(matching_method=method)
        results = benchmark.compare_references(extracted_refs, ground_truth_refs, common_fields)
        
        for field, result in results.items():
            comparison[field][method] = result.f1
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("F1-Score Comparison Across Matching Methods")
    print(f"{'='*80}\n")
    
    header = f"{'Field':<15} " + "".join(f"{m.capitalize():<12}" for m in methods)
    print(header)
    print("-" * len(header))
    
    for field in common_fields:
        row = f"{field:<15} "
        for method in methods:
            f1 = comparison[field].get(method, 0.0)
            row += f"{f1:<12.2f}"
        print(row)
    
    # Best method per field
    print("\nRecommendations:")
    for field in common_fields:
        best_method = max(comparison[field].items(), key=lambda x: x[1])
        print(f"  {field}: {best_method[0]} (F1={best_method[1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Run extraction benchmarks")
    parser.add_argument("--extracted", "-e", required=True, help="Path to extracted references JSON")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare matching methods")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_matching_methods(args.extracted, args.ground_truth)
    else:
        run_benchmark_on_files(args.extracted, args.ground_truth, args.output)


if __name__ == "__main__":
    # Example: Run with command line arguments
    # python run_benchmark.py -e extracted.json -g ground_truth.json
    
    # Or run with default example files if available
    import sys
    if len(sys.argv) == 1:
        print("Usage: python run_benchmark.py -e <extracted.json> -g <ground_truth.json>")
        print("\nExample:")
        print("  python run_benchmark.py -e An_LLM-Based_Framework_for_Synthetic_Data_Generation_references_parsed.json -g ground_truth.json")
        print("\nOptions:")
        print("  -c, --compare    Compare all matching methods and show recommendations")
    else:
        main()
