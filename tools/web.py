import requests
import hashlib
import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from ddgs import DDGS
import config
from .subagent import sub_agent_task

def fetch_url_content(url: str) -> str:
    """Extract raw text content from a web page URL."""
    try:
        soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).text, 'lxml')
        for script in soup(["script", "style", "nav", "footer"]): script.extract()
        return soup.get_text(separator=' ', strip=True)[:4000]
    except Exception as e: return f"Scraping error: {e}"

def search_ddg(query: str) -> str:
    """Search the live internet for up-to-date facts and news."""
    try:
        with DDGS() as ddgs:
            res = [f"Title: {r['title']}\nSnippet: {r['body']}" for r in ddgs.text(query, max_results=3)]
            return "\n\n".join(res) if res else "No results found."
    except Exception as e: return f"Search error: {e}"

def search_wikipedia(subject: str) -> str:
    """Search Wikipedia for biographical, historical, and factual background."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(subject)}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        return res.json().get('extract', 'No summary.') if res.status_code == 200 else "Wiki page not found."
    except Exception as e: return f"Wiki error: {e}"

def search_imdb(query: str) -> str:
    """Search IMDb for movies, TV shows, and actors."""
    return search_ddg(f"site:imdb.com {query}")

def web_research(query: str) -> str:
    """Search the web for a topic and return the 3 most relevant sources (title + link) plus one
    synthesized 2-3 paragraph summary answering the query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"Search error: {e}"

    if not results: return "No results found."

    sources, source_texts = [], []
    for i, r in enumerate(results, 1):
        title = r.get('title', 'Untitled')
        link = r.get('href') or r.get('link') or r.get('url', '')
        sources.append(f"{i}. {title}\n   {link or 'No link available'}")

        page_text = fetch_url_content(link) if link.startswith("http") else ""
        if page_text and "error" not in page_text.lower():
            source_texts.append(f"[Source {i}: {title}]\n{page_text[:2000]}")
        else:
            source_texts.append(f"[Source {i}: {title}]\n{r.get('body', 'No preview available.')}")

    combined = "\n\n".join(source_texts)
    try:
        res = config.client.chat(model=config.SUB_MODEL_NAME, messages=[
            {"role": "system", "content": (
                "You are a research summarizer. Using ONLY the provided sources, write a 2-3 paragraph "
                f"summary that directly answers this query: '{query}'. Synthesize across all sources rather "
                "than listing them one by one. No fluff, no meta-commentary about the sources themselves."
            )},
            {"role": "user", "content": combined[:8000]}
        ], options={"num_predict": 500, "temperature": 0.05})
        summary = res.message.content.strip()
    except Exception as e:
        summary = f"[Summary generation error: {e}]"

    return "Top sources:\n" + "\n".join(sources) + "\n\nSummary:\n" + summary

def scrape_and_summarize_url(url: str, goal: str) -> str:
    """Delegates webpage reading to a fast 0.5B sub-agent to summarize information based on a goal, then saves it to memory."""
    raw_text = fetch_url_content(url)
    if "error" in raw_text.lower(): return raw_text
    
    try:
        res = config.client.chat(model=config.SUB_MODEL_NAME, messages=[
            {"role": "system", "content": "You are a concise research sub-agent. Extract facts from the text that directly answer the Goal. No fluff."},
            {"role": "user", "content": f"Goal: {goal}\n\nText:\n{raw_text[:8000]}"}
        ], options={"num_predict": 400, "temperature": 0.1})
        summary = res.message.content.strip()
        
        if emb := config.get_ollama_embedding(summary):
            doc_id = hashlib.md5(summary.encode()).hexdigest()
            config.kb_collection.add(ids=[f"sub_{doc_id}"], embeddings=[emb], documents=[summary], metadatas=[{"source": url, "goal": goal}])
        return f"Sub-Agent Summary:\n{summary}\n[Saved to DB]"
    except Exception as e: return f"Sub-agent error: {e}"

def ingest_url_to_knowledge_base(url: str) -> str:
    """Permanently embed raw web URL text to vector DB."""
    if "error" in (text := fetch_url_content(url).lower()): return text
    chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
    count = 0
    for i, c in enumerate(chunks):
        doc_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
        if not config.kb_collection.get(ids=[doc_id])['ids']:
            config.kb_collection.add(ids=[doc_id], embeddings=[config.get_ollama_embedding(c)], documents=[c], metadatas=[{"source": url}])
            count += 1
    return f"Ingested {count} chunks from {url}."

def crawl_and_ingest_domain(start_url: str, max_pages: int = 3) -> str:
    """Map out website domains to knowledge base."""
    visited, queue, log = set(), [start_url], []
    while queue and len(visited) < max_pages:
        if (url := queue.pop(0)) in visited: continue
        visited.add(url)
        if "Ingested" in ingest_url_to_knowledge_base(url): log.append(f"Ingested: {url}")
        try:
            soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).text, 'lxml')
            queue.extend([u for link in soup.find_all('a', href=True) if urlparse(u := urljoin(url, link['href']).split('#')[0]).netloc == urlparse(start_url).netloc and u not in visited])
        except Exception: pass
        time.sleep(1)
    return "\n".join(log) + f"\nCrawl complete. Pages processed: {len(visited)}."
