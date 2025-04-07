import os
import asyncio
import aiohttp
import trafilatura
from urllib.parse import urlparse, urljoin
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm
import requests
from lxml import etree
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

VECTOR_STORE_PATH = "data/vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DELAY_BETWEEN_REQUESTS = 2  # seconds

HEADERS = {
    "User-Agent": "LocalGovGPT-Crawler/1.0 (contact)"
}


def parse_sitemap(url, max_urls=100):
    headers = {"User-Agent": "LocalGovGPT-Crawler/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        tree = etree.fromstring(response.content)
        root_tag = tree.tag.lower()

        urls = []

        # Case 1: sitemap index (multiple linked sitemaps)
        if "sitemapindex" in root_tag:
            sitemap_urls = [loc.text for loc in tree.findall(".//{*}loc")]
            print(f"[📦] Sitemap index found with {len(sitemap_urls)} linked sitemaps")
            for sm_url in sitemap_urls:
                try:
                    sub_resp = requests.get(sm_url, headers=headers, timeout=10)
                    sub_resp.raise_for_status()
                    sub_tree = etree.fromstring(sub_resp.content)
                    urls += [loc.text for loc in sub_tree.findall(".//{*}loc")]
                    if len(urls) >= max_urls:
                        break
                except Exception as e:
                    print(f"[!] Failed to fetch sub-sitemap: {sm_url} — {e}")

        # Case 2: regular sitemap
        elif "urlset" in root_tag:
            urls = [loc.text for loc in tree.findall(".//{*}loc")]

        return urls[:max_urls]

    except Exception as e:
        print(f"[!] Failed to parse sitemap at {url}: {e}")
        return []


def get_domain_root(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_same_domain(url, root_domain):
    return url.startswith(root_domain)


def clean_links(base_url, html):


    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full_url = urljoin(base_url, href)
        if is_same_domain(full_url, get_domain_root(base_url)):
            links.add(full_url.split("#")[0])  # strip fragments

    return links


async def fetch(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=10) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def get_seed_urls_from_homepage(home_url, max_seeds=20):
    print(f"[🌱] Extracting seed URLs from: {home_url}")
    headers = {"User-Agent": "LocalGovGPT-Crawler/1.0"}
    try:
        response = requests.get(home_url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"[!] Failed to load homepage for seeds: {e}")
        return [home_url]

    soup = BeautifulSoup(html, "html.parser")
    root = get_domain_root(home_url)
    found = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue

        full_url = urljoin(home_url, href)
        if not is_same_domain(full_url, root):
            continue

        path = urlparse(full_url).path
        if path in ["/", ""] or "?" in path or "#" in path:
            continue

        # Shallow internal URLs only
        if path.count("/") <= 3:
            found.add(full_url.split("#")[0])

    seeds = list(found)
    seeds = sorted(seeds)[:max_seeds]
    print(f"[🌿] Found {len(seeds)} seed URLs")
    return seeds if seeds else [home_url]


def extract_text_from_pdf(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"[!] Failed to extract PDF: {url} — {e}")
        return None


async def crawl_site(start_url, max_pages, min_sitemap_urls):
    visited = set()
    texts = []
    failed = []

    root_domain = get_domain_root(start_url)
    sitemap_url = urljoin(root_domain, "/sitemap.xml")

    # Try sitemap
    crawl_urls = parse_sitemap(sitemap_url, max_urls=max_pages)
    used_sitemap = len(crawl_urls) >= min_sitemap_urls

    if used_sitemap:
        print(f"[🧭] Using sitemap: {sitemap_url} ({len(crawl_urls)} URLs)")
    else:
        print(f"[🔄] Sitemap too short or missing — using smart seeds")
        crawl_urls = get_seed_urls_from_homepage(start_url)

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=min(len(crawl_urls), max_pages), desc=f"Crawling {start_url}")
        to_visit = set(crawl_urls)

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()
            if url in visited:
                continue

            html = await fetch(session, url)
            text = None
            if not html:
                if url.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(url)
                    if not text:
                        failed.append((url, "empty_pdf"))
                        continue
                else:
                    failed.append((url, "no_html"))
                    continue
            else:
                text = trafilatura.extract(html)
                if not text and url.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(url)

            if not text:
                print(f"[⚠️] Failed to extract content from: {url}")
                failed.append((url, "no_extract"))
            else:
                texts.append((url, text))

            visited.add(url)

            # Only expand internal links if we're not using sitemap
            if not used_sitemap and html:
                new_links = clean_links(url, html)
                to_visit.update(new_links - visited)

            pbar.update(1)
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

        pbar.close()

    # Save logs
    save_crawl_log(texts, start_url)
    save_failed_log(failed, start_url)

    print(f"[✅] Finished crawling {start_url}. {len(texts)} pages extracted, {len(failed)} failed.")
    return texts




def embed_and_save(pages, batch_size=100):
    print("[🔢] Preparing documents for embedding...")
    documents = [Document(page_content=text, metadata={"source": url}) for url, text in pages]

    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    all_vectors = []
    all_metas = []

    print(f"[🧠] Embedding {len(split_docs)} chunks in batches of {batch_size}...")

    for i in tqdm(range(0, len(split_docs), batch_size)):
        batch = split_docs[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        metas = [doc.metadata for doc in batch]

        try:
            vectors = embeddings.embed_documents(texts)
            all_vectors.extend(vectors)
            all_metas.extend(metas)
        except Exception as e:
            print(f"[!] Failed embedding batch {i}-{i+batch_size}: {e}")

    print("[💾] Saving vector store...")
    vectordb = FAISS.from_embeddings(all_vectors, all_metas, embeddings)
    vectordb.save_local("data/vector_store")
    print("[✅] Done.")



def load_site_list(file_path="sites_for_crawling.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def save_crawl_log(pages, base_url):
    os.makedirs("logs", exist_ok=True)

    # Clean domain name for filename
    domain = urlparse(base_url).netloc.replace(".", "_")
    filename = f"logs/pages_scraped_{domain}.txt"

    with open(filename, "w") as f:
        for url, _ in pages:
            f.write(url + "\n")

    print(f"[📝] Saved crawl log: {filename}")

def save_failed_log(failed_urls, base_url):
    if not failed_urls:
        return

    os.makedirs("logs", exist_ok=True)
    domain = urlparse(base_url).netloc.replace(".", "_")
    filename = f"logs/pages_failed_{domain}.txt"

    with open(filename, "w") as f:
        for url, reason in failed_urls:
            f.write(f"{url}  # {reason}\n")

    print(f"[🧾] Saved failed crawl log: {filename}")


async def main():
    all_pages = []
    site_list = load_site_list()

    for site in site_list:
        pages = await crawl_site(site, max_pages=100, min_sitemap_urls=50)
        all_pages.extend(pages)
        #save_crawl_log(pages, site)

    embed_and_save(all_pages)


if __name__ == "__main__":
    asyncio.run(main())
