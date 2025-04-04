import requests
import trafilatura
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": "LocalGovGPT (https://github.com/oliverbendix/localgovgpt-nz)"
}

def fetch_and_clean_url(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        downloaded = trafilatura.extract(response.text)
        if downloaded:
            return downloaded
        else:
            print(f"[!] Failed to extract readable content from: {url}")
            return None

    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
        return None

def fetch_multiple(urls, delay=2):
    results = []
    for url in urls:
        print(f"[+] Fetching {url}")
        text = fetch_and_clean_url(url)
        if text:
            results.append({"url": url, "text": text})
        time.sleep(delay)  # Be nice: pause between requests
    return results
