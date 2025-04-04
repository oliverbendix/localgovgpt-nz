import os

from utils.scraper import fetch_multiple

urls = [
    "https://www.aucklandcouncil.govt.nz/report-problem",
    "https://services.wellington.govt.nz/report/"
]

pages = fetch_multiple(urls)

os.makedirs("data/fetched", exist_ok=True)  # ✅ Creates folder if it doesn't exist

# Optional: save to disk for inspection or chunking later
for i, page in enumerate(pages):
    filename = f"data/fetched/page_{i+1}.txt"
    with open(filename, "w") as f:
        f.write(f"Source: {page['url']}\n\n")
        f.write(page["text"])

