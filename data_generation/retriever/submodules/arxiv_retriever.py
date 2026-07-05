
from shlex import quote

import requests

rate_limit_interval = 3 #seconds
retries = 3

def download_pdf(arxiv_id: str, directory: str, filename: str|None = None):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    if filename is None:
        filename = f"{arxiv_id}.pdf"
    else:
        # Ensure filename has .pdf extension
        if not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"
    response = None
    for _ in range(retries):
        response = requests.get(url, stream=True, timeout=rate_limit_interval)
        response.raise_for_status()
        if response is not None: break
    if response is None:
        print(f"[ArxivRetriever] Failed to download PDF after {retries} attempts.")
        return
    
    try:
        with open(f"{directory}/{filename}", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except IOError as e:
        print(f"[ArxivRetriever] Error writing PDF to file: {e}")
        return

    return f"{directory}/{filename}"