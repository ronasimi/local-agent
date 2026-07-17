import os
import glob
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import config

def get_weather(location: str) -> str:
    """Find current weather conditions."""
    try:
        safe_location = requests.utils.quote(location)
        res = requests.get(f"https://wttr.in/{safe_location}?format=%l:+%C+(%c),+%t", timeout=5)
        return res.text.strip() if res.status_code == 200 else "Weather unavailable."
    except Exception as e: return f"Weather error: {e}"

def read_file(filename: str) -> str:
    """Read a text file from the workspace."""
    path = os.path.join(config.WORKSPACE_DIR, os.path.basename(filename))
    if not os.path.exists(path): return f"File '{filename}' not found."
    with open(path, 'r', encoding='utf-8') as f: return f.read()

def write_file(filename: str, content: str) -> str:
    """Write text to a workspace file."""
    os.makedirs(config.WORKSPACE_DIR, exist_ok=True)
    with open(os.path.join(config.WORKSPACE_DIR, os.path.basename(filename)), 'w', encoding='utf-8') as f:
        f.write(content.strip())
    return f"Success: Wrote to '{filename}'."

def list_workspace_files(pattern: str = "*") -> str:
    """Search workspace file names. Pattern can be '*.txt' or '*'."""
    files = [os.path.relpath(f, config.WORKSPACE_DIR) for f in glob.glob(os.path.join(config.WORKSPACE_DIR, pattern), recursive=True) if os.path.isfile(f)]
    return "\n".join(files) if files else "No files found."

def get_system_time() -> str:
    """Verify host target date and time."""
    return f"Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def get_date_info(date_str: str) -> str:
    """Determine the day of the week for a specific date (YYYY-MM-DD)."""
    try: return f"{date_str} was a {datetime.strptime(date_str, '%Y-%m-%d').strftime('%A')}."
    except Exception as e: return f"Date format error: {e}"

def calculate_age(birthdate_str: str) -> str:
    """Calculate exact age in years and days based on a birthdate (YYYY-MM-DD)."""
    try:
        bd = datetime.strptime(birthdate_str, "%Y-%m-%d")
        age = relativedelta(datetime.now(), bd)
        days = (datetime.now() - (bd + relativedelta(years=age.years))).days
        return f"Age: {age.years} years, {days} days."
    except Exception as e: return f"Age calc error: {e}"

def query_knowledge_base(search_term: str) -> str:
    """Query local vector memory."""
    if config.kb_collection.count() == 0: return "Knowledge base empty."
    res = config.kb_collection.query(query_embeddings=[config.get_ollama_embedding(search_term)], n_results=3)
    return "\n\n---\n\n".join([f"[Source: {res['metadatas'][0][i].get('source', 'DB')}]\n{doc}" for i, doc in enumerate(res['documents'][0])]) if res['documents'][0] else "No info found."
