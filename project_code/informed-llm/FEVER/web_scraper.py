import time
import random
import requests
from bs4 import BeautifulSoup

# ── CONFIG ───────────────────────────────────────────────────────────────
MAX_RESULTS     = 3
MIN_DELAY       = 1.0    # seconds
MAX_DELAY       = 2.0    # seconds
MAX_RETRIES     = 5
INITIAL_BACKOFF = 1.0    # seconds

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) "
    "Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.0.0",
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://duckduckgo.com/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}

session = requests.Session()

def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    Polite DuckDuckGo scraping with:
      • User-Agent rotation
      • full header fingerprinting
      • randomized delays
      • exponential back-off on 429/503
    """
    url    = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}
    backoff = INITIAL_BACKOFF

    for _ in range(MAX_RETRIES):
        headers = BASE_HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        try:
            resp = session.post(url, data=params, headers=headers, timeout=10)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 503):
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2
                continue
            raise
    else:
        # If all retries fail, raise last exception
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    snippets = [
        tag.get_text(strip=True)
        for tag in soup.select("a.result__snippet", limit=max_results)
    ]

    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    return "\n".join(snippets)
