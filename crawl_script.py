#@title Download PDF + TeX from paper's title

import requests
import arxiv
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_filename(name):
    """Cleans a string to be safe for use as a filename."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

def download_file_from_url(url, download_dir, filename=None):
    """Downloads a file from a given URL to a specified directory."""
    try:
        os.makedirs(download_dir, exist_ok=True)

        if filename is None:
            # Try to infer filename from URL
            filename = url.split('/')[-1]
            if '.' not in filename: # Add common extension if not present
                filename += '.tar.gz'

        filepath = os.path.join(download_dir, clean_filename(filename))

        logging.info(f"Attempting to download from {url} to {filepath}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded: {filepath}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading from URL {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return False

def download_from_arxiv(paper_title, download_dir):
    """Searches for a paper on arXiv and downloads its source files."""
    try:
        # Ensure the download directory exists
        os.makedirs(download_dir, exist_ok=True)

        # Use the arxiv library to search for the paper title
        # Using triple quotes for the title in the query to handle potential internal quotes better
        search = arxiv.Search(query=f'ti:"{paper_title}"', max_results=1)
        client = arxiv.Client()
        results = client.results(search)
        first_result = next(results, None)

        if first_result:
            cleaned_title = clean_filename(paper_title)

            # Download PDF
            pdf_path = os.path.join(download_dir, f"{cleaned_title}.pdf")
            first_result.download_pdf(dirpath=download_dir, filename=f"{cleaned_title}.pdf")
            logging.info(f"Downloaded PDF to: {pdf_path}")

            # Download TeX source
            # source_path = os.path.join(download_dir, f"{cleaned_title}_source.tar.gz")
            # first_result.download_source(dirpath=download_dir, filename=f"{cleaned_title}_source.tar.gz")
            # logging.info(f"Downloaded TeX source to: {source_path}")

            src_url = 'https://arxiv.org/src/' + first_result.get_short_id()
            download_file_from_url(src_url, download_dir, filename=f"{cleaned_title}_source.tar.gz")

            return True
        else:
            logging.warning(f"Could not find '{paper_title}' on arXiv.")
            return False
    except Exception as e:
        logging.error(f"Error downloading from arXiv for '{paper_title}': {e}")
        return False
