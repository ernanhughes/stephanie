import requests
from bs4 import BeautifulSoup
import csv
import json
import re

COLLECTION_URL = "https://huggingface.co/collections/The-Great-Genius/skynet-66366061cc7af105efb7e0ca"

# URL of the Hugging Face collection
url = "https://huggingface.co/collections/The-Great-Genius/skynet-66366061cc7af105efb7e0ca"

# Fetch the page content
response = requests.get(url)
html = response.text

# Parse with BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Extract all hrefs
links = [a.get("href") for a in soup.find_all("a", href=True)]

# Match ArXiv-style links
arxiv_links = [link for link in links if re.search(r"paper", link)]

# Deduplicate
arxiv_links = list(set(arxiv_links))

# Output
print("Found ArXiv Papers:")
for link in arxiv_links:
    print(link)
# Convert to full ArXiv links
arxiv_links1 = [f"https://arxiv.org/abs{path.replace('/papers', '')}" for path in arxiv_links]

# Output
for link in arxiv_links1:
    print(link)

