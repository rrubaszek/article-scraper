import requests
import re
import trafilatura

from collections import deque
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class Scraper:
    def extract(self, url: str, source: str) -> str:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(downloaded)
        meta = trafilatura.extract_metadata(downloaded)

        if not text or len(text.split()) < 200:
            return None

        return {
            "url": url,
            "title": meta.title if meta else None,
            "date": meta.date if meta else None,
            "text": text,
            "source": source
        }

    def get_links(self, page_url: str, regex: re.Pattern, headers: dict | None = None, blocklist: list[str] | None = None) -> list[str]:
        r = requests.get(page_url, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # normalize
            full = urljoin(page_url, href)

            # Filter out english articles
            if blocklist and any(block in full for block in blocklist):
                continue
    
            if re.search(regex, full):
                links.add(full)

        return list(links)

    def crawl(self, seed_urls, regex: re.Pattern, headers: dict | None = None, blocklist: list[str] | None = None, max_pages: int = 5000):
        q = deque(seed_urls)
        seen = set()
        articles = set()

        while q and len(seen) < max_pages:
            url = q.popleft()

            if url in seen:
                continue
            seen.add(url)

            try:
                links = self.get_links(url, headers=headers, regex=regex, blocklist=blocklist)
            except:
                continue

            for link in links:
                # article heuristic (important)
                if "/artykul/" in link or link.count("/") > 5:
                    articles.add(link)
                else:
                    q.append(link)

        return list(articles)