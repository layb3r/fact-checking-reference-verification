#!/usr/bin/env python3

import json
import os
import tarfile
import logging
import shutil
import subprocess
import re
from pathlib import Path
from crawl_utils import download_from_arxiv, download_source_from_arxiv_id
import fitz 

# Try to import bibtexparser, provide helpful error if not available
try:
    import bibtexparser
except ImportError:
    print("Error: bibtexparser is not installed. Install it with:")
    print("pip install bibtexparser")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import subprocess
import os
import re
import logging

def find_bib_file(directory):
    """Recursively search for .bib files in a directory."""
    bib_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bib'):
                bib_files.append(os.path.join(root, file))
    return bib_files


def parse_bib_to_json(bib_filepath):
    """Parse a .bib file and convert to JSON format."""
    try:
        with open(bib_filepath, 'r', encoding='utf-8') as bibfile:
            bib_database = bibtexparser.load(bibfile)
        
        # Convert to list format, filtering out non-reference entries
        # bibtexparser automatically separates @String definitions from entries
        # entries only contains actual references like @article, @inproceedings, etc.
        references = []
        for entry in bib_database.entries:
            # Double-check that this is an actual reference entry (has ENTRYTYPE)
            if 'ENTRYTYPE' in entry:
                references.append(entry)
        
        logging.info(f"Parsed {len(references)} references from {bib_filepath}")
        return references
    except Exception as e:
        logging.error(f"Error parsing .bib file {bib_filepath}: {e}")
        return []


def extract_tar_file(tar_path, extract_to):
    """Extract a tar.gz or tar.xz file to specified directory."""
    try:
        logging.info(f"Extracting {tar_path} to {extract_to}")
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Successfully extracted to {extract_to}")
        return True
    except Exception as e:
        logging.error(f"Error extracting {tar_path}: {e}")
        return False


def find_tex_files(directory):
    """Recursively search for .tex files in a directory."""
    tex_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    return tex_files


def find_main_tex_file(directory):
    """Find the main .tex file (usually main.tex or files with \\documentclass)."""
    tex_files = find_tex_files(directory)
    
    # First, look for common main file names
    for tex_file in tex_files:
        basename = os.path.basename(tex_file).lower()
        if basename in ['main.tex', 'paper.tex', 'manuscript.tex']:
            return tex_file
    
    # If not found, search for files containing \documentclass
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5KB
                if '\\documentclass' in content:
                    return tex_file
        except Exception:
            continue
    
    # If still not found, return the first .tex file
    return tex_files[0] if tex_files else None


def compile_latex(tex_file):
    """
    Compile a .tex file to generate .bbl file.
    Returns the path to the .bbl file if successful, None otherwise.
    """
    if not tex_file or not os.path.exists(tex_file):
        logging.warning("No .tex file found to compile")
        return None
    
    tex_dir = os.path.dirname(tex_file)
    tex_basename = os.path.basename(tex_file)
    tex_name = os.path.splitext(tex_basename)[0]
    bbl_file = os.path.join(tex_dir, f"{tex_name}.bbl")
    
    # Check if .bbl already exists
    if os.path.exists(bbl_file):
        logging.info(f".bbl file already exists: {bbl_file}")
        return bbl_file
    
    logging.info(f"Compiling LaTeX file: {tex_file}")
    
    try:
        # Run pdflatex
        logging.info("Running pdflatex...")
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_basename],
            cwd=tex_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check if .aux file was created (needed for bibtex)
        aux_file = os.path.join(tex_dir, f"{tex_name}.aux")
        if not os.path.exists(aux_file):
            logging.warning("No .aux file created, cannot run bibtex")
            return None
        
        # Run bibtex
        logging.info("Running bibtex...")
        result = subprocess.run(
            ['bibtex', tex_name],
            cwd=tex_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if .bbl file was created
        if os.path.exists(bbl_file):
            logging.info(f"Successfully generated .bbl file: {bbl_file}")
            return bbl_file
        else:
            logging.warning("bibtex did not generate .bbl file")
            return None
            
    except subprocess.TimeoutExpired:
        logging.error("LaTeX compilation timed out")
        return None
    except FileNotFoundError:
        logging.error("pdflatex or bibtex not found. Please install TeX Live or similar.")
        return None
    except Exception as e:
        logging.error(f"Error during LaTeX compilation: {e}")
        return None


def parse_bbl_file(bbl_filepath):
    """
    Parse a .bbl file and extract bibliography entries.
    Returns a list of dictionaries containing cite_key and bbl_text.
    """
    if not os.path.exists(bbl_filepath):
        logging.warning(f".bbl file not found: {bbl_filepath}")
        return []
    
    try:
        with open(bbl_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract bibliography entries using regex
        # Look for \bibitem patterns
        entries = []
        
        # Pattern 1: \bibitem{cite_key}
        # Pattern 2: \bibitem[label]{cite_key}
        pattern = r'\\bibitem(?:\[.*?\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for cite_key, bbl_text in matches:
            # Clean up the text
            bbl_text = bbl_text.strip()
            if bbl_text:
                entries.append({
                    'cite_key': cite_key,
                    'bbl_text': bbl_text
                })
        
        logging.info(f"Parsed {len(entries)} entries from .bbl file")
        return entries
        
    except Exception as e:
        logging.error(f"Error parsing .bbl file {bbl_filepath}: {e}")
        return []


def find_citations_in_tex(directory):
    """
    Find all citation commands in .tex files.
    Returns a dictionary mapping cite_key to list of citation instances.
    Each instance contains: file_path, line_number, full_line, citation_command
    """
    citations = {}
    
    tex_files = find_tex_files(directory)
    
    # Patterns for various citation commands
    # \cite{key}, \citep{key}, \citet{key}, \cite[text]{key}, etc.
    citation_pattern = r'\\cite[a-z]*(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}'
    
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith('%'):
                    continue
                
                # Find all citation commands in this line
                matches = re.finditer(citation_pattern, line)
                for match in matches:
                    cite_keys_str = match.group(1)
                    # Handle multiple keys in one cite command (e.g., \cite{key1,key2})
                    cite_keys = [k.strip() for k in cite_keys_str.split(',')]
                    
                    for cite_key in cite_keys:
                        if cite_key not in citations:
                            citations[cite_key] = []
                        
                        citations[cite_key].append({
                            'file_path': tex_file,
                            'line_number': line_num,
                            'full_line': line.strip(),
                            'citation_command': match.group(0),
                            'lines_context': lines  # Store all lines for context extraction
                        })
        
        except Exception as e:
            logging.warning(f"Error reading {tex_file}: {e}")
            continue
    
    logging.info(f"Found citations for {len(citations)} unique cite keys")
    return citations


def extract_citation_context(citation_info, context_sentences=2):
    """
    Extract claim text and surrounding context from a citation.
    
    Args:
        citation_info: Dictionary with file_path, line_number, full_line, lines_context
        context_sentences: Number of sentences to include before/after citation
    
    Returns:
        Dictionary with 'claim_text' and 'surrounding_context'
    """
    try:
        lines = citation_info['lines_context']
        line_idx = citation_info['line_number'] - 1  # Convert to 0-based index
        
        # Get a window of lines around the citation
        window_size = 5  # Lines before and after
        start_idx = max(0, line_idx - window_size)
        end_idx = min(len(lines), line_idx + window_size + 1)
        
        # Combine lines into text
        text_window = ' '.join(lines[start_idx:end_idx])
        
        # Clean up LaTeX commands and extra whitespace
        text_window = re.sub(r'%.*', '', text_window)  # Remove comments
        text_window = re.sub(r'\\label\{[^}]+\}', '', text_window)  # Remove labels
        text_window = re.sub(r'\s+', ' ', text_window).strip()  # Normalize whitespace
        
        # Find the sentence containing the citation
        # Split by sentence boundaries (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text_window)
        
        # Find which sentence contains the citation command
        citation_cmd = citation_info['citation_command']
        claim_sentence = None
        claim_idx = 0
        
        for i, sent in enumerate(sentences):
            if citation_cmd in sent or citation_info['full_line'] in sent:
                claim_sentence = sent
                claim_idx = i
                break
        
        if claim_sentence is None:
            # Fallback: use the full line
            claim_sentence = citation_info['full_line']
            claim_idx = len(sentences) // 2
        
        # Extract surrounding context
        start_context = max(0, claim_idx - context_sentences)
        end_context = min(len(sentences), claim_idx + context_sentences + 1)
        surrounding = ' '.join(sentences[start_context:end_context])
        
        # Replace citation commands with [CITATION] marker
        # Pattern matches \cite variants like \cite{}, \citep{}, \citet{}, with optional arguments
        citation_pattern = r'\\cite[a-z]*(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^}]+\}'
        claim_sentence = re.sub(citation_pattern, '[CITATION]', claim_sentence)
        surrounding = re.sub(citation_pattern, '[CITATION]', surrounding)
        
        # Clean up multiple spaces that may result from replacement
        claim_sentence = re.sub(r'\s+', ' ', claim_sentence).strip()
        surrounding = re.sub(r'\s+', ' ', surrounding).strip()
        
        return {
            'claim_text': claim_sentence,
            'surrounding_context': surrounding
        }
    
    except Exception as e:
        logging.warning(f"Error extracting context: {e}")
        # Also add marker to fallback
        fallback_text = citation_info.get('full_line', '')
        citation_pattern = r'\\cite[a-z]*(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^}]+\}'
        fallback_text = re.sub(citation_pattern, '[CITATION]', fallback_text)
        return {
            'claim_text': fallback_text,
            'surrounding_context': fallback_text
        }


def bib_entry_to_metadata(bib_entry):
    """
    Convert a bibtex entry to CitationMetadata format.
    
    Args:
        bib_entry: Dictionary from bibtexparser
    
    Returns:
        Dictionary matching CitationMetadata schema
    """
    # Extract authors
    authors = []
    if 'author' in bib_entry:
        author_str = bib_entry['author']
        # Split by 'and'
        author_list = re.split(r'\s+and\s+', author_str)
        for author in author_list:
            # Clean up LaTeX formatting
            author = re.sub(r'[{}]', '', author)
            author = author.strip()
            if author:
                authors.append(author)
    
    # Extract year
    year = None
    if 'year' in bib_entry:
        try:
            year = int(bib_entry['year'])
        except (ValueError, TypeError):
            year = None
    
    # Extract identifiers
    doi = bib_entry.get('doi', None)
    arxiv_id = None
    url = bib_entry.get('url', None)
    
    # Try to extract arXiv ID from various fields
    if 'eprint' in bib_entry and 'arxiv' in bib_entry.get('archiveprefix', '').lower():
        arxiv_id = bib_entry['eprint']
    elif 'arxiv' in bib_entry:
        arxiv_id = bib_entry['arxiv']
    elif url and 'arxiv.org' in url:
        # Extract from URL
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9.]+)', url)
        if match:
            arxiv_id = match.group(1)
    
    # Extract arXiv ID from journal field (e.g., "arXiv preprint arXiv:2009.09761")
    if not arxiv_id and 'journal' in bib_entry:
        journal = bib_entry['journal']
        if 'arxiv' in journal.lower():
            # Look for pattern like "arXiv:XXXX.XXXXX" or "arXiv XXXX.XXXXX"
            match = re.search(r'arXiv[:\s]+([0-9]{4}\.[0-9]+)', journal, re.IGNORECASE)
            if match:
                arxiv_id = match.group(1)
    
    return {
        'title': bib_entry.get('title', None),
        'authors': authors if authors else [],
        'venue': bib_entry.get('booktitle') or bib_entry.get('journal') or bib_entry.get('publisher', None),
        'year': year,
        'identifiers': {
            'doi': doi,
            'arxiv_id': arxiv_id,
            'url': url
        }
    }


def is_valid_claim_text(claim_text):
    """
    Check if claim text is valid (sentence-like, not LaTeX commands/tables/figures).
    
    Args:
        claim_text: The claim text string
    
    Returns:
        Boolean indicating if the claim is valid
    """
    if not claim_text or len(claim_text.strip()) < 20:
        return False
    
    # Must contain the [CITATION] marker
    if '[CITATION]' not in claim_text:
        return False
    
    # Filter out text with too many LaTeX commands
    latex_markers = [
        r'\\begin\{',
        r'\\end\{',
        r'\\centering',
        r'\\caption',
        r'\\includegraphics',
        r'\\hline',
        r'\\toprule',
        r'\\midrule',
        r'\\bottomrule',
    ]
    
    for marker in latex_markers:
        if re.search(marker, claim_text):
            return False
    
    # Check if it contains some actual words (not just LaTeX/symbols)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', claim_text)
    if len(words) < 5:
        return False
    
    return True


def create_dataset_instance(cite_key, bib_entry, citation_info):
    """
    Create a complete dataset instance according to the schema.
    
    Args:
        cite_key: Citation key string
        bib_entry: Dictionary from bibtexparser
        citation_info: Dictionary with citation location and context
    
    Returns:
        Dictionary matching DatasetInstance schema, or None if invalid
    """
    # Extract context
    context = extract_citation_context(citation_info)
    
    # Validate claim text
    if not is_valid_claim_text(context['claim_text']):
        return None
    
    # Convert bib entry to metadata
    metadata = bib_entry_to_metadata(bib_entry)
    
    # Create ground truth (all positive for published papers)
    ground_truth = {
        'true_existence': 1,  # Reference exists
        'true_hallucination_category': None,  # No hallucination
        'true_alignment': 0,  # Fully supported (assumption for published papers)
        'expert_rationale': 'Citation from published paper, assumed to be correct and fully supported.'
    }
    
    return {
        'claim_text': context['claim_text'],
        'surrounding_context': context['surrounding_context'],
        'citation_metadata': metadata,
        'true_outputs': ground_truth
    }


def sort_references_by_bbl(references, bbl_entries):
    """
    Sort references according to the order they appear in .bbl file.
    Also adds compiled_bbl field to each reference by compiling all bibitems together.
    References not found in .bbl will be placed at the end.
    """
    # Create a dictionary for fast lookup
    ref_dict = {ref.get('ID', ref.get('id', '')): ref for ref in references}
    
    # Sort references by bbl order and add compiled_bbl
    sorted_refs = []
    
    # Process references in bbl order
    for entry in bbl_entries:
        cite_key = entry['cite_key']
        
        if cite_key in ref_dict:
            ref_copy = ref_dict[cite_key].copy()
            
            sorted_refs.append(ref_copy)
    
    # Add any references not in bbl at the end (without compiled_bbl)
    sorted_ref_ids = {ref.get('ID', ref.get('id', '')) for ref in sorted_refs}
    for ref in references:
        ref_id = ref.get('ID', ref.get('id', ''))
        if ref_id not in sorted_ref_ids:
            sorted_refs.append(ref)
    
    return sorted_refs


def process_id(arxiv_id, temp_dir):
    """
    Process a single paper and generate dataset instances:
    1. Download source from arXiv id
    2. Extract source archive
    3. Find and parse .bib file
    4. Find citations in .tex files
    5. Create dataset instances for each citation
    6. Return dataset instances
    """
    logging.info(f"Processing: {arxiv_id}")
    
    # Create temporary directory
    paper_temp_dir = os.path.join(temp_dir, arxiv_id.replace('/', '_'))
    os.makedirs(paper_temp_dir, exist_ok=True)
    
    try:
        # Download source from arxiv id
        logging.info("Step 1: Downloading from arXiv...")
        download_result = download_source_from_arxiv_id(arxiv_id, paper_temp_dir)
        if not download_result:
            logging.error(f"Failed to download {arxiv_id}")
            return None
        
        # Find the downloaded source file
        source_file = os.path.join(paper_temp_dir, f"{arxiv_id.replace('/', '_')}_source.tar.gz")
        
        # Check if source file exists
        if not os.path.exists(source_file):
            logging.error(f"Source file not found: {source_file}")
            return None
        
        # Extract source archive
        logging.info("Step 2: Extracting source archive...")
        extract_dir = os.path.join(paper_temp_dir, "extracted")
        if not extract_tar_file(source_file, extract_dir):
            logging.error(f"Failed to extract source for {arxiv_id}")
            return None
        
        # Find .bib files
        logging.info("Step 3: Finding .bib files...")
        bib_files = find_bib_file(extract_dir)
        
        if not bib_files:
            logging.warning(f"No .bib files found for {arxiv_id}")
            return None
        
        logging.info(f"Found {len(bib_files)} .bib file(s): {[os.path.basename(f) for f in bib_files]}")
        
        # Parse ALL .bib files and combine references
        references = {}  # Dictionary mapping cite_key to bib_entry
        
        for bib_file in bib_files:
            file_refs = parse_bib_to_json(bib_file)
            for ref in file_refs:
                ref_id = ref.get('ID', ref.get('id', ''))
                if ref_id:
                    references[ref_id] = ref
        
        logging.info(f"Total unique references from all .bib files: {len(references)}")
        
        if not references:
            logging.warning(f"No references found in .bib files for {arxiv_id}")
            return None
        
        # Step 4: Find citations in .tex files
        logging.info("Step 4: Finding citations in .tex files...")
        citations = find_citations_in_tex(extract_dir)
        
        if not citations:
            logging.warning(f"No citations found in .tex files for {arxiv_id}")
            return None
        
        logging.info(f"Found citations for {len(citations)} cite keys")
        
        # Step 5: Create dataset instances
        logging.info("Step 5: Creating dataset instances...")
        dataset_instances = []
        
        for cite_key, citation_list in citations.items():
            # Check if we have the bib entry for this cite key
            if cite_key not in references:
                logging.debug(f"No bib entry found for cite key: {cite_key}")
                continue
            
            bib_entry = references[cite_key]
            
            # Limit to at most 2 instances per reference
            instances_for_this_ref = 0
            max_instances_per_ref = 2
            
            # Create an instance for each citation occurrence
            for citation_info in citation_list:
                if instances_for_this_ref >= max_instances_per_ref:
                    break
                
                try:
                    instance = create_dataset_instance(cite_key, bib_entry, citation_info)
                    if instance is not None:  # Only add valid instances
                        dataset_instances.append(instance)
                        instances_for_this_ref += 1
                except Exception as e:
                    logging.warning(f"Error creating instance for {cite_key}: {e}")
                    continue
        
        logging.info(f"Created {len(dataset_instances)} dataset instances")
        
        # Create result data
        result = {
            "arxiv_id": arxiv_id,
            "arxiv_url": "https://arxiv.org/abs/" + arxiv_id,
            "num_instances": len(dataset_instances),
            "instances": dataset_instances
        }
        
        logging.info(f"Successfully processed: {arxiv_id}\n")
        return result
        
    finally:
        # Clean up temporary files
        logging.info("Cleaning up temporary files...")
        try:
            if os.path.exists(paper_temp_dir):
                shutil.rmtree(paper_temp_dir)
                logging.info(f"Cleaned up temporary directory: {paper_temp_dir}")
        except Exception as e:
            logging.warning(f"Error cleaning up temporary files: {e}")

def main():
    """
    Main function to process papers and create dataset.
    Reads paper list from CSV and generates dataset instances.
    """
    import argparse
    import csv
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Generate citation dataset from arXiv papers')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to CSV file containing paper list')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: auto-generated)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of papers to process')
    parser.add_argument('--temp-dir', type=str, default='temp_processing',
                       help='Temporary directory for downloads')
    parser.add_argument('--arxiv-col', type=str, default='arxiv_id',
                       help='Column name containing arXiv IDs')
    
    args = parser.parse_args()
    
    # Read paper list from CSV
    logging.info(f"Reading paper list from: {args.input}")
    arxiv_ids = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.arxiv_col in row and row[args.arxiv_col]:
                    arxiv_ids.append(row[args.arxiv_col].strip())
                    if args.limit and len(arxiv_ids) >= args.limit:
                        break
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return
    
    logging.info(f"Found {len(arxiv_ids)} papers to process")
    
    if not arxiv_ids:
        logging.error("No arXiv IDs found in input file")
        return
    
    # Create temporary directory
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each paper
    all_instances = []
    successful_papers = 0
    failed_papers = 0
    
    for i, arxiv_id in enumerate(arxiv_ids, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing paper {i}/{len(arxiv_ids)}: {arxiv_id}")
        logging.info(f"{'='*60}")
        
        try:
            result = process_id(arxiv_id, temp_dir)
            
            if result and result['instances']:
                all_instances.extend(result['instances'])
                successful_papers += 1
                logging.info(f"✓ Successfully processed {arxiv_id}: {result['num_instances']} instances")
            else:
                failed_papers += 1
                logging.warning(f"✗ No instances generated for {arxiv_id}")
        
        except Exception as e:
            failed_papers += 1
            logging.error(f"✗ Error processing {arxiv_id}: {e}")
            continue
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"citation_dataset_{timestamp}.json"
    
    # Save dataset
    logging.info(f"\n{'='*60}")
    logging.info("Dataset Generation Summary")
    logging.info(f"{'='*60}")
    logging.info(f"Total papers processed: {len(arxiv_ids)}")
    logging.info(f"Successful: {successful_papers}")
    logging.info(f"Failed: {failed_papers}")
    logging.info(f"Total dataset instances: {len(all_instances)}")
    
    # Save to JSON
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'creation_date': datetime.now().isoformat(),
                    'num_papers': successful_papers,
                    'num_instances': len(all_instances),
                    'source': args.input
                },
                'instances': all_instances
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"\n✓ Dataset saved to: {args.output}")
        
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")
    
    # Clean up temp directory
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Error cleaning up temp directory: {e}")

if __name__ == "__main__":
    main()