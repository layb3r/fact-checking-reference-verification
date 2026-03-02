#!/usr/bin/env python3

import json
import os
import tarfile
import logging
import shutil
from pathlib import Path
from crawl_script import download_from_arxiv, clean_filename

# Try to import bibtexparser, provide helpful error if not available
try:
    import bibtexparser
except ImportError:
    print("Error: bibtexparser is not installed. Install it with:")
    print("pip install bibtexparser")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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


def process_paper(paper_name, output_base_dir, temp_dir):
    """
    Process a single paper:
    1. Download PDF and source from arXiv
    2. Extract source archive
    3. Find and parse .bib file
    4. Save organized output
    5. Clean up temporary files
    """
    logging.info(f"Processing: {paper_name}")
    
    cleaned_name = clean_filename(paper_name)
    
    # Create directories
    paper_output_dir = os.path.join(output_base_dir, cleaned_name)
    paper_temp_dir = os.path.join(temp_dir, cleaned_name)
    os.makedirs(paper_output_dir, exist_ok=True)
    os.makedirs(paper_temp_dir, exist_ok=True)
    
    # Download from arXiv
    logging.info("Step 1: Downloading from arXiv...")
    pdf_url = download_from_arxiv(paper_name, paper_temp_dir)
    if not pdf_url:
        logging.error(f"Failed to download {paper_name}")
        # Clean up output directory if it was created
        if os.path.exists(paper_output_dir):
            shutil.rmtree(paper_output_dir)
            logging.info(f"Cleaned up output directory: {paper_output_dir}")
        return False
    
    # Find the downloaded source file
    source_file = os.path.join(paper_temp_dir, f"{cleaned_name}_source.tar.gz")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        logging.error(f"Source file not found: {source_file}")
        # Clean up output directory
        if os.path.exists(paper_output_dir):
            shutil.rmtree(paper_output_dir)
            logging.info(f"Cleaned up output directory: {paper_output_dir}")
        return False
    
    # Extract source archive
    logging.info("Step 2: Extracting source archive...")
    extract_dir = os.path.join(paper_temp_dir, "extracted")
    if not extract_tar_file(source_file, extract_dir):
        logging.error(f"Failed to extract source for {paper_name}")
        # Clean up output directory
        if os.path.exists(paper_output_dir):
            shutil.rmtree(paper_output_dir)
            logging.info(f"Cleaned up output directory: {paper_output_dir}")
        return False
    
    # Find .bib files
    logging.info("Step 3: Finding .bib files...")
    bib_files = find_bib_file(extract_dir)
    
    if not bib_files:
        logging.warning(f"No .bib files found for {paper_name}")
        references = []
        if os.path.exists(paper_output_dir):
            shutil.rmtree(paper_output_dir)
            logging.info(f"Cleaned up output directory: {paper_output_dir}")
        return False
    else:
        logging.info(f"Found {len(bib_files)} .bib file(s): {[os.path.basename(f) for f in bib_files]}")
        
        # Parse ALL .bib files and combine references
        references = []
        seen_ids = set()  # Track unique reference IDs to avoid duplicates
        
        for bib_file in bib_files:
            file_refs = parse_bib_to_json(bib_file)
            # Add only unique references (check by ID)
            for ref in file_refs:
                ref_id = ref.get('ID', ref.get('id', ''))
                if ref_id and ref_id not in seen_ids:
                    references.append(ref)
                    seen_ids.add(ref_id)
                elif not ref_id:
                    # If no ID, add anyway (rare case)
                    references.append(ref)
        
        logging.info(f"Total unique references from all .bib files: {len(references)}")
    
    # Save organized output
    logging.info("Step 4: Saving output...")
    
    # Create output JSON with PDF URL and references
    output_data = {
        "pdf_url": pdf_url,
        "references": references
    }
    
    # Save data as JSON
    output_json = os.path.join(paper_output_dir, "references.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved data: {output_json} (PDF URL + {len(references)} references)")
    
    # Clean up temporary files
    logging.info("Step 5: Cleaning up temporary files...")
    try:
        shutil.rmtree(paper_temp_dir)
        logging.info(f"Cleaned up temporary directory: {paper_temp_dir}")
    except Exception as e:
        logging.warning(f"Error cleaning up temporary files: {e}")
    
    logging.info(f"Successfully processed: {paper_name}\n")
    return True


def main():
    """Main function to process all papers from JSON file."""
    # Configuration
    input_json = "/content/nips_500.json"
    output_dir = "output/papers"
    temp_dir = "temp/downloads"
    
    # Create base directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Read input JSON
    logging.info(f"Reading papers from {input_json}")
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except Exception as e:
        logging.error(f"Error reading {input_json}: {e}")
        return
    
    logging.info(f"Found {len(papers)} papers to process\n")
    
    # Process each paper
    successful = 0
    failed = 0
    
    for i, paper in enumerate(papers, 1):
        paper_name = paper.get('name', '') if isinstance(paper, dict) else paper
        if not paper_name:
            logging.warning(f"Paper {i} has no name, skipping")
            failed += 1
            continue
        
        logging.info(f"[{i}/{len(papers)}] Processing: {paper_name[:60]}...")
        
        try:
            if process_paper(paper_name, output_dir, temp_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logging.error(f"Unexpected error processing {paper_name}: {e}")
            failed += 1
    
    # Summary
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing complete!")
    logging.info(f"Successful: {successful}/{len(papers)}")
    logging.info(f"Failed: {failed}/{len(papers)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"{'='*80}")
    
    # Clean up temp directory if empty
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            logging.info("Cleaned up empty temp directory")
    except Exception as e:
        logging.warning(f"Could not remove temp directory: {e}")


if __name__ == "__main__":
    main()

# create .bbl:
# !pdflatex main.tex
# !bibtex main