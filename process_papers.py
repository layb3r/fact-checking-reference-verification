#!/usr/bin/env python3

import json
import os
import tarfile
import logging
import shutil
import subprocess
import re
from pathlib import Path
from crawl_script import download_from_arxiv, clean_filename
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

def compile_all_bibitems(bbl_entries, temp_dir="temp"):
    if not bbl_entries:
        return {}
    
    # Use absolute paths to avoid confusion
    base_dir = os.getcwd()
    temp_path = os.path.abspath(temp_dir)
    os.makedirs(temp_path, exist_ok=True)
    
    temp_tex = "all_items.tex"
    temp_pdf = "all_items.pdf"
    
    all_bibitems = []
    for entry in bbl_entries:
        # Ensure bbl_text doesn't have naked '%' which comments out the rest of the file
        all_bibitems.append(f"\\bibitem{{{entry['cite_key']}}}\n{entry['bbl_text']}")
    
    latex_template = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{url}
\pagenumbering{gobble}
\hyphenpenalty=10000
\exhyphenpenalty=10000
\sloppy
\begin{document}
\begin{thebibliography}{99}
""" + "\n\n".join(all_bibitems) + r"""
\end{thebibliography}
\end{document}
"""

    try:
        # Write file inside temp_path
        with open(os.path.join(temp_path, temp_tex), "w", encoding="utf-8") as f:
            f.write(latex_template)

        # CRITICAL: Run pdflatex WITH cwd=temp_path
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", temp_tex], 
            cwd=temp_path,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True,
            timeout=30
        )
        
        # Extract text using path relative to temp_path or absolute path
        result = subprocess.check_output(
            ["pdftotext", "-layout", os.path.join(temp_path, temp_pdf), "-"], 
            encoding="utf-8",
            timeout=10
        )

        # print(result)
        
        # ... (rest of your parsing logic remains the same) ...
        compiled_dict = {}
        lines = result.split('\n')
        current_ref_num = None
        current_text = []
        
        for line in lines:
            match = re.match(r'\s*\[(\d+)\]\s*(.*)', line)
            if match:
                if current_ref_num is not None:
                    ref_index = current_ref_num - 1
                    if ref_index < len(bbl_entries):
                        compiled_dict[bbl_entries[ref_index]['cite_key']] = ' '.join(current_text).strip()
                current_ref_num = int(match.group(1))
                current_text = [match.group(2).strip()]
            elif current_ref_num is not None:
                current_text.append(line.strip())

        if current_ref_num is not None:
            ref_index = current_ref_num - 1
            if ref_index < len(bbl_entries):
                compiled_dict[bbl_entries[ref_index]['cite_key']] = ' '.join(current_text).strip()

        return compiled_dict

    except subprocess.CalledProcessError as e:
        logging.error(f"LaTeX Error. Check {os.path.join(temp_path, 'all_items.log')} for details.")
        return {}


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


def sort_references_by_bbl(references, bbl_entries, temp_dir="temp"):
    """
    Sort references according to the order they appear in .bbl file.
    Also adds compiled_bbl field to each reference by compiling all bibitems together.
    References not found in .bbl will be placed at the end.
    """
    # Compile all bibitems at once
    # compiled_dict = compile_all_bibitems(bbl_entries, temp_dir)
    
    # Create a dictionary for fast lookup
    ref_dict = {ref.get('ID', ref.get('id', '')): ref for ref in references}
    
    # Sort references by bbl order and add compiled_bbl
    sorted_refs = []
    
    # Process references in bbl order
    for entry in bbl_entries:
        cite_key = entry['cite_key']
        bbl_text = entry['bbl_text']
        
        if cite_key in ref_dict:
            ref_copy = ref_dict[cite_key].copy()
            
            # Get compiled text from the dictionary
            # if cite_key in compiled_dict:
            #     ref_copy['compiled_bbl'] = compiled_dict[cite_key]
            # else:
            #     # Fallback to raw bbl_text if compilation failed
            #     ref_copy['compiled_bbl'] = bbl_text
            #     logging.warning(f"No compiled text for {cite_key}, using raw text")
            
            sorted_refs.append(ref_copy)
    
    # Add any references not in bbl at the end (without compiled_bbl)
    sorted_ref_ids = {ref.get('ID', ref.get('id', '')) for ref in sorted_refs}
    for ref in references:
        ref_id = ref.get('ID', ref.get('id', ''))
        if ref_id not in sorted_ref_ids:
            sorted_refs.append(ref)
    
    return sorted_refs


def process_paper(paper_name, temp_dir):
    """
    Process a single paper:
    1. Download PDF and source from arXiv
    2. Extract source archive
    3. Find and parse .bib file
    4. Compile LaTeX to get .bbl file
    5. Sort references by .bbl order
    6. Return paper data
    """
    logging.info(f"Processing: {paper_name}")
    
    cleaned_name = clean_filename(paper_name)
    
    # Create temporary directory
    paper_temp_dir = os.path.join(temp_dir, cleaned_name)
    os.makedirs(paper_temp_dir, exist_ok=True)
    
    try:
        # Download from arXiv
        logging.info("Step 1: Downloading from arXiv...")
        pdf_url = download_from_arxiv(paper_name, paper_temp_dir)
        if not pdf_url:
            logging.error(f"Failed to download {paper_name}")
            return None
        
        # Find the downloaded source file
        source_file = os.path.join(paper_temp_dir, f"{cleaned_name}_source.tar.gz")
        
        # Check if source file exists
        if not os.path.exists(source_file):
            logging.error(f"Source file not found: {source_file}")
            return None
        
        # Extract source archive
        logging.info("Step 2: Extracting source archive...")
        extract_dir = os.path.join(paper_temp_dir, "extracted")
        if not extract_tar_file(source_file, extract_dir):
            logging.error(f"Failed to extract source for {paper_name}")
            return None
        
        # Find .bib files
        logging.info("Step 3: Finding .bib files...")
        bib_files = find_bib_file(extract_dir)
        
        if not bib_files:
            logging.warning(f"No .bib files found for {paper_name}")
            return None
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
        
        # Step 4: Compile LaTeX to generate .bbl file
        logging.info("Step 4: Attempting to compile LaTeX to generate .bbl file...")
        bbl_entries = []
        
        main_tex = find_main_tex_file(extract_dir)
        if main_tex:
            logging.info(f"Found main .tex file: {os.path.basename(main_tex)}")
            bbl_file = compile_latex(main_tex)
            
            if bbl_file:
                # Parse the .bbl file
                bbl_entries = parse_bbl_file(bbl_file)
                
                if bbl_entries:
                    logging.info(f"Successfully parsed {len(bbl_entries)} entries from .bbl file")
                    # Sort references and add compiled_bbl to each
                    references = sort_references_by_bbl(references, bbl_entries, paper_temp_dir)[:len(bbl_entries)]
                    logging.info("Sorted references and added compiled_bbl fields")
                else:
                    logging.warning("No entries found in .bbl file")
            else:
                logging.warning("Could not generate .bbl file")
        else:
            logging.warning("No main .tex file found for compilation")
        
        # Create paper data
        paper_data = {
            "paper_name": paper_name,
            "pdf_url": pdf_url,
            "references": references
        }
        
        logging.info(f"Successfully processed: {paper_name}\n")
        return paper_data
        
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
    """Main function to process all papers from JSON file."""
    # Configuration
    input_json = "./data/neurips_data/filtered_nips_5307.json"
    output_json = "output/all_papers.json"
    temp_dir = "temp/downloads"
    
    # Create base directories
    os.makedirs("output", exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Read input JSON
    logging.info(f"Reading papers from {input_json}")
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            papers = json.load(f)[:300]
            # For testing with a single paper:
            # papers = ['Generalized Linear Mode Connectivity for Transformers']
    except Exception as e:
        logging.error(f"Error reading {input_json}: {e}")
        return
    
    logging.info(f"Found {len(papers)} papers to process\n")
    
    # Process each paper and collect results
    all_papers_data = []
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
            paper_data = process_paper(paper_name, temp_dir)
            if paper_data:
                all_papers_data.append(paper_data)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logging.error(f"Unexpected error processing {paper_name}: {e}")
            failed += 1
    
    # Save all papers to a single JSON file
    logging.info(f"\nSaving all papers to {output_json}...")
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_papers_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved {len(all_papers_data)} papers to {output_json}")
    except Exception as e:
        logging.error(f"Error saving output JSON: {e}")
    
    # Summary
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing complete!")
    logging.info(f"Successful: {successful}/{len(papers)}")
    logging.info(f"Failed: {failed}/{len(papers)}")
    logging.info(f"Output file: {output_json}")
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