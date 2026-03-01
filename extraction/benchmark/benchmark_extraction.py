#!/usr/bin/env python3
"""
Extraction Benchmarking System
Evaluates extraction quality by comparing extracted fields with ground truth.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import difflib


@dataclass
class MatchingResult:
    """Result of matching a single field"""
    field: str
    matches: int = 0
    mismatches: int = 0
    missing: int = 0
    
    @property
    def precision(self) -> float:
        total_predicted = self.matches + self.mismatches
        return self.matches / total_predicted if total_predicted > 0 else 0.0
    
    @property
    def recall(self) -> float:
        total_actual = self.matches + self.missing
        return self.matches / total_actual if total_actual > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.matches + self.mismatches + self.missing
        return self.matches / total if total > 0 else 0.0


class FieldMatcher:
    """Implements various matching methods for field comparison"""
    
    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Normalize text for soft matching"""
        if not text:
            return ""
        # Lowercase, remove punctuation, collapse whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def strict_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
        """Exact byte-for-byte match"""
        return extracted == ground_truth if extracted and ground_truth else False
    
    @staticmethod
    def soft_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
        """Normalized comparison"""
        if not extracted or not ground_truth:
            return False
        return FieldMatcher.normalize_text(extracted) == FieldMatcher.normalize_text(ground_truth)
    
    @staticmethod
    def levenshtein_match(extracted: Optional[str], ground_truth: Optional[str], threshold: float = 0.80) -> bool:
        """Edit distance with threshold"""
        if not extracted or not ground_truth:
            return False
        
        # Use difflib for simple implementation
        similarity = difflib.SequenceMatcher(None, extracted.lower(), ground_truth.lower()).ratio()
        return similarity >= threshold
    
    @staticmethod
    def ratcliff_match(extracted: Optional[str], ground_truth: Optional[str], threshold: float = 0.95) -> bool:
        """Ratcliff/Obershelp matching"""
        if not extracted or not ground_truth:
            return False
        
        similarity = difflib.SequenceMatcher(None, extracted.lower(), ground_truth.lower()).ratio()
        return similarity >= threshold
    
    @staticmethod
    def author_match(extracted: List[str], ground_truth: List[str], threshold: float = 0.75) -> bool:
        """Jaccard similarity for author lists (last names)"""
        if not extracted or not ground_truth:
            return False
        
        # Extract last names (simple heuristic: last word)
        def get_last_name(author: str) -> str:
            return author.strip().split()[-1].lower() if author else ""
        
        extracted_last = {get_last_name(a) for a in extracted if a}
        gt_last = {get_last_name(a) for a in ground_truth if a}
        
        if not extracted_last or not gt_last:
            return False
        
        intersection = len(extracted_last & gt_last)
        union = len(extracted_last | gt_last)
        jaccard = intersection / union if union > 0 else 0.0
        
        return jaccard >= threshold
    
    @staticmethod
    def year_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
        """Exact year match (extract year from date strings)"""
        if not extracted or not ground_truth:
            return False
        
        # Extract 4-digit year
        def extract_year(text: str) -> Optional[str]:
            match = re.search(r'\b(19|20)\d{2}\b', str(text))
            return match.group(0) if match else None
        
        ext_year = extract_year(extracted)
        gt_year = extract_year(ground_truth)
        
        return ext_year == gt_year if ext_year and gt_year else False
    
    @staticmethod
    def identifier_match(extracted: Optional[str], ground_truth: Optional[str]) -> bool:
        """Identifier matching (DOI, arXiv) with normalization"""
        if not extracted or not ground_truth:
            return False
        
        # Normalize: lowercase, remove common prefixes
        def normalize_id(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r'^(https?://)?(doi\.org/|arxiv\.org/abs/)', '', text)
            # Remove arXiv version
            text = re.sub(r'v\d+$', '', text)
            return text
        
        return normalize_id(extracted) == normalize_id(ground_truth)


class ExtractionBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, matching_method: str = "soft"):
        """
        Args:
            matching_method: One of ['strict', 'soft', 'levenshtein', 'ratcliff']
        """
        self.matching_method = matching_method
        self.matcher = FieldMatcher()
        self.results: Dict[str, MatchingResult] = {}
    
    def match_field(self, field: str, extracted: any, ground_truth: any) -> bool:
        """Match a single field using configured method"""
        # Handle special fields
        if field == "authors":
            # Convert to list if needed
            ext_authors = extracted if isinstance(extracted, list) else [extracted] if extracted else []
            gt_authors = ground_truth if isinstance(ground_truth, list) else [ground_truth] if ground_truth else []
            return self.matcher.author_match(ext_authors, gt_authors)
        
        elif field == "year":
            return self.matcher.year_match(str(extracted) if extracted else None, 
                                          str(ground_truth) if ground_truth else None)
        
        elif field in ["doi", "arxiv_id", "isbn"]:
            return self.matcher.identifier_match(str(extracted) if extracted else None,
                                                str(ground_truth) if ground_truth else None)
        
        # Text fields: use configured matching method
        ext_str = str(extracted) if extracted else None
        gt_str = str(ground_truth) if ground_truth else None
        
        if self.matching_method == "strict":
            return self.matcher.strict_match(ext_str, gt_str)
        elif self.matching_method == "soft":
            return self.matcher.soft_match(ext_str, gt_str)
        elif self.matching_method == "levenshtein":
            return self.matcher.levenshtein_match(ext_str, gt_str)
        elif self.matching_method == "ratcliff":
            return self.matcher.ratcliff_match(ext_str, gt_str)
        else:
            return self.matcher.soft_match(ext_str, gt_str)
    
    def compare_references(self, extracted_refs: List[Dict], ground_truth_refs: List[Dict], 
                          fields: List[str]) -> Dict[str, MatchingResult]:
        """
        Compare extracted references with ground truth.
        
        Args:
            extracted_refs: List of extracted reference dicts
            ground_truth_refs: List of ground truth reference dicts
            fields: List of field names to compare (e.g., ['title', 'authors', 'year'])
        
        Returns:
            Dict mapping field names to MatchingResult objects
        """
        # Initialize results
        results = {field: MatchingResult(field=field) for field in fields}
        
        # Align references by index (assumes same order)
        for i, gt_ref in enumerate(ground_truth_refs):
            if i >= len(extracted_refs):
                # Missing extraction
                for field in fields:
                    if gt_ref.get(field):
                        results[field].missing += 1
                continue
            
            ext_ref = extracted_refs[i]
            
            # Compare each field
            for field in fields:
                gt_value = gt_ref.get(field)
                ext_value = ext_ref.get(field)
                
                # Handle missing ground truth
                if not gt_value:
                    continue
                
                # Handle missing extraction
                if not ext_value:
                    results[field].missing += 1
                    continue
                
                # Perform matching
                if self.match_field(field, ext_value, gt_value):
                    results[field].matches += 1
                else:
                    results[field].mismatches += 1
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate formatted report of benchmark results"""
        if not self.results:
            return "No results available. Run compare_references first."
        
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"Extraction Benchmark Results (Method: {self.matching_method})")
        report.append(f"{'='*80}\n")
        
        # Table header
        header = f"{'Field':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'Matches':<8} {'Mismatches':<11} {'Missing':<8}"
        report.append(header)
        report.append("-" * len(header))
        
        # Field rows
        for field, result in self.results.items():
            row = (f"{field:<15} "
                  f"{result.precision:<10.2f} "
                  f"{result.recall:<10.2f} "
                  f"{result.f1:<10.2f} "
                  f"{result.accuracy:<10.2f} "
                  f"{result.matches:<8} "
                  f"{result.mismatches:<11} "
                  f"{result.missing:<8}")
            report.append(row)
        
        # Overall metrics
        avg_precision = sum(r.precision for r in self.results.values()) / len(self.results)
        avg_recall = sum(r.recall for r in self.results.values()) / len(self.results)
        avg_f1 = sum(r.f1 for r in self.results.values()) / len(self.results)
        avg_accuracy = sum(r.accuracy for r in self.results.values()) / len(self.results)
        
        report.append("-" * len(header))
        overall = (f"{'Overall (Macro)':<15} "
                  f"{avg_precision:<10.2f} "
                  f"{avg_recall:<10.2f} "
                  f"{avg_f1:<10.2f} "
                  f"{avg_accuracy:<10.2f}")
        report.append(overall)
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON file"""
        results_dict = {
            "matching_method": self.matching_method,
            "fields": {}
        }
        
        for field, result in self.results.items():
            results_dict["fields"][field] = {
                "precision": round(result.precision, 4),
                "recall": round(result.recall, 4),
                "f1": round(result.f1, 4),
                "accuracy": round(result.accuracy, 4),
                "matches": result.matches,
                "mismatches": result.mismatches,
                "missing": result.missing
            }
        
        # Add overall metrics
        results_dict["overall"] = {
            "macro_precision": round(sum(r.precision for r in self.results.values()) / len(self.results), 4),
            "macro_recall": round(sum(r.recall for r in self.results.values()) / len(self.results), 4),
            "macro_f1": round(sum(r.f1 for r in self.results.values()) / len(self.results), 4),
            "macro_accuracy": round(sum(r.accuracy for r in self.results.values()) / len(self.results), 4)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)


def run_benchmark_example():
    """Example usage with sample data"""
    # Sample ground truth
    ground_truth = [
        {
            "title": "Attention is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": "2017",
            "venue": "NeurIPS",
            "doi": "10.5555/3295222.3295349"
        }
    ]
    
    # Sample extracted data (with minor variations)
    extracted = [
        {
            "title": "Attention Is All You Need",  # Different case
            "authors": ["A Vaswani", "N Shazeer"],  # Abbreviated
            "year": "2017",
            "venue": "Neural Information Processing Systems",  # Full name
            "doi": "10.5555/3295222.3295349"
        }
    ]
    
    # Run benchmark with different methods
    fields = ["title", "authors", "year", "venue", "doi"]
    
    for method in ["strict", "soft", "levenshtein"]:
        print(f"\n{'='*80}")
        print(f"Testing with {method.upper()} matching")
        print(f"{'='*80}")
        
        benchmark = ExtractionBenchmark(matching_method=method)
        benchmark.compare_references(extracted, ground_truth, fields)
        print(benchmark.generate_report())


if __name__ == "__main__":
    run_benchmark_example()
