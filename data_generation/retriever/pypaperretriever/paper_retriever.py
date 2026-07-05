"""Retrieve and download scientific papers from various sources."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from glob import glob
from urllib.parse import urljoin, urlparse

import fitz as pymupdf
import requests
from bs4 import BeautifulSoup
from typing import Self

from .utils import decode_doi, doi_to_pmid, encode_doi, entrez_efetch, pmid_to_doi


class PaperRetriever:
    """Find and download scientific papers.

    The class queries several services (Unpaywall, PubMed Central, Crossref and
    optionally Sci-Hub) to locate a PDF for a given DOI or PMID.

    Args:
        email (str): Email address used for API requests.
        doi (str, optional): Digital Object Identifier of the paper.
        pmid (str, optional): PubMed identifier of the paper.
        allow_scihub (bool, optional): Whether to query Sci-Hub as a fallback.
        download_directory (str, optional): Directory where PDFs are stored.
        filename (str, optional): Custom filename for the downloaded PDF.
        override_previous_attempt (bool, optional): Overwrite existing downloads.

    Attributes:
        doi (str): DOI encoded for safe file paths.
        pmid (str | None): PubMed ID of the paper.
        pdf_urls (list[str]): Candidate URLs pointing to PDF files.
        filepath (str | None): Path to the downloaded PDF if successful.
        is_downloaded (bool): ``True`` if the PDF has been retrieved.
        is_oa (bool): ``True`` if the paper is open access.
        on_scihub (bool): ``True`` if the PDF was found on Sci-Hub.
    """

    def __init__(self, email, doi=None, pmid=None, allow_scihub=False, download_directory='PDFs', filename=None, override_previous_attempt=False):
        self.email = email
        if not doi and not pmid:
            raise ValueError("Either a DOI or PMID must be provided")
        if not doi and pmid:
            doi = pmid_to_doi(pmid, email)
        self.doi = encode_doi(doi)
        self.pmid = pmid
        self.allow_scihub = allow_scihub
        self.is_oa = False
        self.on_scihub = False
        # Indicates whether the paper/resource was found (metadata or known location),
        # even if the PDF download was unsuccessful.
        self.found = False
        self.pdf_urls = []
        self.is_downloaded = False
        self.filepath = None
        self.override_previous_attempt = override_previous_attempt
        self.download_directory = download_directory
        self.filename = filename
        self.user_agents = [
            "Dalvik/1.6.0 (Linux; U; Android 6.2.2; J6 Build/JDQ39)",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.104 AOL/9.8 AOLBuild/4346.1012.US Safari/537.36",
            "Dalvik/2.1.0 (Linux; U; Android 5.1.1; K3DX-V5G Build/LMY47V)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36,gzip(gfe)",
            "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:83.0) Gecko/20100101 Firefox/83.0",
            "Mozilla/5.0 (Linux; Android 5.0; P01M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 9; MiTV-AXSO0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 10; SM-A107F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.116 Mobile Safari/537.36 OPR/55.2.2719.50740",
            "Mozilla/5.0 (Linux; Android 9; Redmi S2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.62 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 11; Pixel 2 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36",        
        ]
 
    def download(self) -> Self:
        """Find and download the paper.

        The method queries multiple services for PDF links and attempts to
        download the first accessible file. Metadata about the attempt is stored
        alongside the PDF.

        Returns:
            Self: This instance.

        """
        if not self.override_previous_attempt:
            self._look_for_previous_download()
            if self.is_downloaded:
                return self
        
        self.check_open_access()
        self.check_pubmed_central_access()
        self.check_crossref_access(decode_doi(self.doi))
        if len(self.pdf_urls) > 0:
            # We discovered candidate PDF URLs (open-access); paper exists even if download later fails
            self.found = True
            print("[PyPaperRetriever] Found Open-Access PDF link(s). Attempting download...")
            if self._download_pdf():
                return self
        if self.allow_scihub:
            self.pdf_urls = []
            self.check_scihub_access()
            if len(self.pdf_urls) > 0:
                print("[PyPaperRetriever] Found PDF on Sci-Hub. Attempting download...")
                self._download_pdf()
                if self.is_downloaded:
                    return self
            else:
                print(f"[PyPaperRetriever] No PDFs found for {decode_doi(self.doi)}")
                
        else:
            print(f"[PyPaperRetriever] No Open-Access PDF found for {decode_doi(self.doi)}. Sci-Hub access is disabled.")
        self._download_pdf()  # Just to create JSON sidecar
        return self
    
    def check_open_access(self) -> Self:
        """Check Unpaywall for open-access availability.

        Updates ``pdf_urls`` with any links returned by the API and sets
        ``is_oa`` if open-access links are found.

        Returns:
            Self: This instance.

        """

        url = f"https://api.unpaywall.org/v2/{decode_doi(self.doi)}?email={self.email}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            pdf_urls = [None, None, None, None]
            pdf_locations = [loc.get("url_for_pdf") for loc in data.get("oa_locations", []) if loc.get("url_for_pdf")]
            pdf_urls[:len(pdf_locations)] = pdf_locations[:4]
            pdf_urls = [url for url in pdf_urls if url]

            pubmed_europe_info = next((
                (loc.get("url").split("?")[0], loc.get("url").split("pmc")[-1].split("/")[0])
                for loc in data.get("oa_locations", [])
                if "europepmc.org/articles/pmc" in loc.get("url", "")
            ), (None, None))
            pubmed_europe_url, pmcid = pubmed_europe_info # Not used in current implementation

            if len(pdf_urls) > 0:
                self.is_oa = True
                self.pdf_urls += pdf_urls
            # Unpaywall returned metadata for the DOI, so the paper exists.
            self.found = True
            return self

        else:
            print("error", f"Unpaywall API request failed with status code {response.status_code}")
            return self
        
    def check_pubmed_central_access(self) -> Self:
        """Check whether the article is available in PubMed Central.

        Any discovered PDF links are appended to ``pdf_urls``.

        Returns:
            Self: This instance.

        """
        pmc_id = None
        pmid_or_id = self.pmid if self.pmid else doi_to_pmid(decode_doi(self.doi), self.email)
        if pmid_or_id is None:
            print(f"No PMID/ID available to check PMC access for doi {self.doi}")
            return self
        records = entrez_efetch(self.email, str(pmid_or_id))
        try:
            id_list = records['PubmedArticle'][0]['PubmedData']['ArticleIdList']
            for element in id_list:
                if element.attributes.get('IdType') == 'pmc':
                    pmc_id = str(element)

        except Exception as e:
            print(f"Error processing while checking PMC access for id {id}: {e}")

        if pmc_id is not None:

            article_link = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/'

            response = requests.get(article_link, headers={"User-Agent": random.choice(self.user_agents)})

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find PDF links
                pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
                pdf_links = [f"{article_link}{pdf_link}" if pdf_link.startswith('/') else pdf_link for pdf_link in pdf_links]
                pdf_links = list(set(pdf_links))
                for link in pdf_links:
                    self.pdf_urls.append(link)
                # Found content on PMC; mark as existing
                if pdf_links:
                    self.found = True
            else:
                print(f"Failed to fetch the PubMed Central link. Status code: {response.status_code}")

        return self

    def check_crossref_access(self, doi: str) -> Self:
        """Query Crossref for PDF links.

        Args:
            doi (str): DOI to query.

        Returns:
            Self: This instance.

        """
        base_url = "https://api.crossref.org/works/"
        full_url = f"{base_url}{doi}"
        urls = []
        pdf_urls = []
        
        try:
            response = requests.get(full_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                primary_url = data.get('message', {}).get('URL')
                if primary_url:
                    urls.append(primary_url)
                doi_link = f"https://doi.org/{doi}"
                urls.append(doi_link)
                for link_entry in data.get('message', {}).get('link', []):
                    pdf_link = link_entry.get('URL')
                    if pdf_link:
                        pdf_urls.append(pdf_link)

            for url in urls:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    }
                    response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                    if response.status_code == 406:
                        headers['Accept'] = '*/*'
                        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)

                    if response.status_code == 200:
                        final_url = response.url  # The final resolved URL after redirects
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        pdf_links = set()

                        # 1. Extract PDF links from <a> tags
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            if href.lower().endswith('.pdf'):
                                absolute_url = urljoin(final_url, href)
                                pdf_links.add(absolute_url)
                        
                        # 2. Extract PDF links from JavaScript
                        for script in soup.find_all('script'):
                            if script.string:
                                # Regex to find patterns like window.open('/path/to/file.pdf') or href = "/path/to/file.pdf"
                                matches = re.findall(r'''(?:window\.open|href\s*=\s*)\(['"]([^'"]+\.pdf)['"]\)''', script.string, re.IGNORECASE)
                                for match in matches:
                                    absolute_url = urljoin(final_url, match)
                                    pdf_links.add(absolute_url)
                                
                                # Another regex pattern based on the example provided
                                matches = re.findall(r'''location\s*=\s*['"]([^'"]+\.pdf)['"]''', script.string, re.IGNORECASE)
                                for match in matches:
                                    absolute_url = urljoin(final_url, match)
                                    pdf_links.add(absolute_url)

                        # 3. Optionally, search for direct links in data attributes or other patterns
                        # Example: data-pdf-url="/path/to/file.pdf"
                        data_pdf_urls = re.findall(r'data-pdf-url=["\']([^"\']+\.pdf)["\']', response.text, re.IGNORECASE)
                        for match in data_pdf_urls:
                            absolute_url = urljoin(final_url, match)
                            pdf_links.add(absolute_url)

                        # Remove any invalid URLs (optional)
                        valid_pdf_links = set()
                        for link in pdf_links:
                            parsed = urlparse(link)
                            if parsed.scheme in ['http', 'https']:
                                valid_pdf_links.add(link)

                        if valid_pdf_links:
                            for link in valid_pdf_links:
                                pdf_urls.append(link)

                    else:
                        if response.status_code != 406:
                            print(f"Failed to access URL: {url} with status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Error accessing URL: {url}")
                    print(e)

            final_pdf_urls = list(set(pdf_urls))
            for link in final_pdf_urls:
                self.pdf_urls.append(link)
        
        except requests.exceptions.RequestException as e:
            print("Something went wrong while trying to access Crossref API")
            print(e)
        return self

    def check_scihub_access(self) -> Self:
        """Search Sci-Hub mirrors for the paper.

        Introduces small delays and rotates user agents to reduce the likelihood
        of being blocked.

        Returns:
            Self: This instance.

        """
        mirror_list = [
            "https://sci-hub.vn",
            # "https://sci-hub.se",
            # "https://sci-hub.wf",
            # "https://sci-hub.st",
            # "https://sci-hub.ru",
        ]
        urls = [f"{mirror}/{decode_doi(self.doi)}" for mirror in mirror_list]

        for i, url in enumerate(urls):
            time.sleep(random.randint(3, 4)) # Delay between requests, avoids being blocked
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            }
            print(f"[PyPaperRetriever] Trying Sci-Hub mirror: {mirror_list[i]}")
            try:
                #try access base url first to check if mirror is up before making the full request with DOI (some mirrors block requests with DOI in URL but allow access to homepage, so this is not a guarantee but can reduce some failed requests)
                r = requests.get(mirror_list[i], headers=headers, timeout=10)
                if r.status_code != 200:
                    print(f"[PyPaperRetriever] Sci-Hub mirror {mirror_list[i]} is not accessible (status code {r.status_code}). Skipping...")
                    continue
                r = requests.get(url, headers=headers, timeout=20)
                if r.status_code == 200:
                    if len(r.text) < 1:
                        print("""We probably got blocked by Sci-Hub for too many requests. 
                            For being a source of free scientific knowledge, they sure are stingy with their bandwidth.
                            Although they don't specify rate limit and have no robots.txt, they still block IPs with too many requests.
                            Try connecting to a different proxy IP with a VPN.""")
                        continue
                    result = self._get_pdf_element(r.text, mirror_list[i])
                    if result == "unavailable":
                        continue
                    elif result:
                        self.on_scihub = True
                        self.pdf_urls.append(result)
                        # Sci-Hub provided a PDF link; mark as found
                        self.found = True
                        break
                else:
                    print(f"[PyPaperRetriever] Sci-Hub mirror returned {r.status_code}: {mirror_list[i]}")
            except requests.RequestException as e:
                print(f"Failed to scrape {url} due to {e}")
                print("If this error includes 'Connection reset by peer', your ISP may be blocking Sci-Hub. Try using a VPN, like ProtonVPN.")
                continue
        return self

    def _download_pdf(self) -> bool:
        """Download the first accessible PDF in ``pdf_urls``.

        Returns:
            bool: ``True`` if a PDF was downloaded successfully.

        """
        file_directory, pdf_path, json_path = self._determine_paths()
        os.makedirs(file_directory, exist_ok=True)

        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }

        if not self.pdf_urls:
            self.filepath = "unavailable"
            self._create_json_sidecar(download_success=False, pdf_filepath=pdf_path, json_filepath=json_path)
            return False

        for pdf_url in self.pdf_urls:
            try:
                response = requests.get(pdf_url, headers=headers, stream=True)
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    # Check if the file is downloaded and not corrupted
                    # Validate the specific file we just wrote
                    self._check_if_downloaded(pdf_path, '.pdf')
                    if self.is_downloaded:
                        self._create_json_sidecar(
                            download_success=True,
                            pdf_filepath=pdf_path,
                            json_filepath=json_path,
                            url=pdf_url,
                        )
                        print(
                            f"[PyPaperRetriever] PDF downloaded successfully to {pdf_path} for {decode_doi(self.doi)} from {pdf_url}"
                        )
                        return True
            except requests.RequestException as e:
                continue

        # If no URLs resulted in a successful download
        self.filepath = "unavailable"
        self._create_json_sidecar(download_success=False, pdf_filepath=pdf_path, json_filepath=json_path)
        print(f"[PyPaperRetriever] Failed to download PDF for {decode_doi(self.doi)}")
        return False

    def _create_json_sidecar(
        self,
        download_success: bool,
        pdf_filepath: str,
        json_filepath: str,
        url: str | None = None,
    ) -> None:
        """Write a JSON sidecar describing the download attempt.

        Args:
            download_success (bool): ``True`` if the PDF was downloaded.
            pdf_filepath (str): Location of the downloaded PDF.
            json_filepath (str): Where to write the sidecar JSON.
            url (str | None): Source URL of the PDF.
        """

        open_access = self.is_oa
        if (url and ('scihub' in url or 'sci-hub' in url)) :
            open_access = False
        info = {
            'doi': decode_doi(self.doi),
            'encoded_doi': self.doi,
            'pmid': self.pmid,
            'id': self.pmid if self.pmid else self.doi,
            'source_url':url,
            'all_urls': self.pdf_urls,
            'download_success': download_success,
            'pdf_filepath': pdf_filepath if download_success else "unavailable",
            'open_access': open_access
        }
        with open(json_filepath, 'w') as f:
            json.dump(info, f, indent=4)

    def _normalize_filename(self, filename: str) -> str:
        """Ensure a custom filename has a .pdf extension."""
        if not filename.lower().endswith(".pdf"):
            return f"{filename}.pdf"
        return filename

    def _determine_paths(self) -> tuple[str, str, str]:
        """Determine output paths for the PDF and its sidecar.

        Returns:
            tuple[str, str, str]: Directory, PDF path and JSON path.

        """
        if self.filename:
            normalized_filename = self._normalize_filename(self.filename)
            file_directory = self.download_directory
            pdf_path = os.path.join(self.download_directory, normalized_filename)
            # Build JSON path by removing .pdf extension and adding .json
            base_filename = normalized_filename[:-4] if normalized_filename.lower().endswith('.pdf') else normalized_filename
            json_path = os.path.join(self.download_directory, f"{base_filename}.json")
        else:
            subdir_name = f"pmid-{self.pmid}" if self.pmid else f"doi-{self.doi}"
            file_directory = os.path.join(self.download_directory, subdir_name)
            filename = f"pmid-{self.pmid}.pdf" if self.pmid else f"doi-{self.doi}.pdf"
            pdf_path = os.path.join(file_directory, filename)
            # Build JSON path by removing .pdf extension and adding .json
            base_filename = filename[:-4] if filename.lower().endswith('.pdf') else filename
            json_path = os.path.join(file_directory, f"{base_filename}.json")

        return file_directory, pdf_path, json_path

    def _check_if_downloaded(self, download_directory_or_path: str, filetype: str = ".pdf") -> Self:
        """Verify that a downloaded file is not corrupted.

        Args:
            download_directory_or_path (str): Directory or file path to inspect.
            filetype (str): Expected file extension.

        Returns:
            Self: This instance.

        """
        # If a specific file path was provided, validate that file only
        if os.path.isfile(download_directory_or_path):
            file_path = download_directory_or_path
            try:
                with pymupdf.open(file_path) as doc:
                    if len(doc) > 0:
                        self.is_downloaded = True
                        self.filepath = file_path
                        return self
            except Exception:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            self.is_downloaded = False
            return self

        # Otherwise treat the argument as a directory and validate PDFs inside it.
        files_with_type = glob(os.path.join(download_directory_or_path, f"*{filetype}"))
        valid_files = []

        for file_path in files_with_type:
            try:
                with pymupdf.open(file_path) as doc:
                    if len(doc) > 0:
                        valid_files.append(file_path)
            except Exception:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        if valid_files:
            # Prefer the most recently modified valid file in the directory
            valid_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            self.is_downloaded = True
            self.filepath = valid_files[0]
        else:
            self.is_downloaded = False

        return self
    
    def _look_for_previous_download(self) -> Self:
        """Check whether a previous download attempt exists.

        Returns:
            Self: This instance with ``is_downloaded`` set if a prior attempt is recorded.

        """
        file_directory, pdf_path, json_path = self._determine_paths()

        if os.path.exists(json_path):
            json_data = json.load(open(json_path))
            self.filepath = json_data.get("pdf_filepath", "unavailable")
            self.is_downloaded = json_data.get("download_success", False)
        else:
            self.is_downloaded = False
            self.filepath = None
        return self

    def _get_pdf_element(self, html_text: str, mirror: str) -> str:
        """Extract the PDF link from a Sci-Hub HTML response.

        Args:
            html_text (str): HTML retrieved from a Sci-Hub mirror.
            mirror (str): Mirror URL used for the request.

        Returns:
            str: Resolved PDF link or ``"unavailable"`` if not found.
        """
        soup = BeautifulSoup(html_text, 'lxml')
        pdf_link = ""
        if soup.find('p', string=re.compile(r"Unfortunately, Sci-Hub doesn't have the requested document", re.I)):
            return "unavailable"

        candidates = []
        for tag, attr in [('embed', 'src'), ('iframe', 'src'), ('object', 'data')]:
            element = soup.find(tag, {attr: True})
            if element and element.has_attr(attr):
                candidates.append(element[attr])

        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if href.lower().endswith('.pdf') or '.pdf?' in href.lower():
                candidates.append(href)

        if not candidates:
            candidates.extend(re.findall(r'(?:https?:)?//[^"\s<>]*?\.pdf(?:\?[^"\s<>]*)?', html_text, re.IGNORECASE))

        for pdf_link_raw in candidates:
            if not pdf_link_raw:
                continue
            pdf_link_raw = pdf_link_raw.strip()
            if pdf_link_raw.startswith('//'):
                pdf_link = 'https:' + pdf_link_raw
            elif pdf_link_raw.startswith('/'):
                pdf_link = f"{mirror}{pdf_link_raw}"
            elif pdf_link_raw.startswith('http://') or pdf_link_raw.startswith('https://'):
                pdf_link = pdf_link_raw
            else:
                pdf_link = urljoin(mirror, pdf_link_raw)

            parsed = urlparse(pdf_link)
            if parsed.scheme in ('http', 'https'):
                return pdf_link

        return pdf_link

def main() -> None:
    """Run the command-line interface.

    The interface accepts DOI or PMID identifiers and downloads the
    corresponding PDFs using :class:`PaperRetriever`.
    """
    parser = argparse.ArgumentParser(description='Download scientific papers automatically.')
    parser.add_argument('--email', required=True, help='Email address for API usage.')
    parser.add_argument('--doi', help='Digital Object Identifier of the paper.')
    parser.add_argument('--pmid', help='PubMed ID of the paper.')
    parser.add_argument('--dwn-dir', default='PDFs', help='Directory to download the PDFs into. Defaults to "PDFs".')
    parser.add_argument('--filename', help='Custom filename for the downloaded PDF.')
    parser.add_argument('--override', action='store_true', help='Override previous download attempts.')
    parser.add_argument('--allow-scihub', choices=['true', 'false'], default='false',
                    help='Allow downloading from Sci-Hub if available (true/false).')

    args = parser.parse_args()
    args.allow_scihub = args.allow_scihub.lower() == 'true' 

    retriever = PaperRetriever(
        email=args.email,
        doi=args.doi,
        pmid=args.pmid,
        download_directory=args.dwn_dir,
        filename=args.filename,
        override_previous_attempt=args.override,
        allow_scihub=args.allow_scihub,
    )

    retriever.download()

