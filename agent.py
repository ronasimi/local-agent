import os, json, sys, time, requests, base64, docker, glob, psutil, pymysql, chromadb, re, hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ollama import Client
from ddgs import DDGS
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

# --- Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_CTX = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"))
MODEL_NAME = "qwen2.5:7b-instruct"  # general instruct model: uses native hermes-style tool_calls reliably
SUB_MODEL_NAME = "qwen2.5-coder:0.5b"     # stays tiny -- it only does narrow extraction/summarization

MEMORY_DIR = os.getenv("MEMORY_DIR", "/app/memory")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/app/workspace")
MEMORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json")
CMD_HISTORY_FILE = os.path.join(MEMORY_DIR, "cmd_history.txt")

# --- Database Credentials ---
# No hardcoded fallbacks: this tool stays disabled (see query_mariadb) until you set these explicitly.
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

client = Client(host=OLLAMA_HOST)

# --- Databases ---
os.makedirs(CHROMA_PATH := os.path.join(MEMORY_DIR, "vector_db"), exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
kb_collection = chroma_client.get_or_create_collection(name="agent_knowledge")
few_shot_collection = chroma_client.get_or_create_collection(name="agent_few_shot")

def copy_to_clipboard(text: str):
    """Copies text to the host clipboard via OSC 52 ANSI escape sequence."""
    sys.stdout.write(f"\033]52;c;{base64.b64encode(text.encode()).decode()}\a")
    sys.stdout.flush()

# ==========================================
# 0.5B SUB-AGENT DELEGATION PROTOCOL
# ==========================================

def sub_agent_task(text: str, instruction: str) -> str:
    """Passes messy or token-heavy text to the 0.5B model for extraction/summarization."""
    if not text or "error" in text.lower() or len(text) < 200: 
        return text 
    try:
        res = client.chat(model=SUB_MODEL_NAME, messages=[
            {"role": "system", "content": f"You are a concise data-extraction sub-agent. {instruction} No fluff."},
            {"role": "user", "content": text[:8000]}
        ], options={"num_predict": 300, "temperature": 0.1})
        return f"[0.5B Summary] {res.message.content.strip()}"
    except Exception as e: 
        return f"[Sub-agent error: {e}]\n<raw_text>\n{text[:500]}\n</raw_text>"

# ==========================================
# 🛠️ AGENT TOOLS (Auto-Parsed by Ollama SDK)
# ==========================================

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
    synthesized 2-3 paragraph summary answering the query. Prefer this over search_ddg whenever the
    user wants sources and a written answer, not just a quick fact."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No results found."

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
        res = client.chat(model=SUB_MODEL_NAME, messages=[
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

def get_weather(location: str) -> str:
    """Find current weather conditions."""
    try:
        safe_location = requests.utils.quote(location)
        res = requests.get(f"https://wttr.in/{safe_location}?format=%l:+%C+(%c),+%t", timeout=5)
        return res.text.strip() if res.status_code == 200 else "Weather unavailable."
    except Exception as e: return f"Weather error: {e}"

def read_file(filename: str) -> str:
    """Read a text file from the workspace."""
    path = os.path.join(WORKSPACE_DIR, os.path.basename(filename))
    if not os.path.exists(path): return f"File '{filename}' not found."
    with open(path, 'r', encoding='utf-8') as f: return f.read()

def write_file(filename: str, content: str) -> str:
    """Write text to a workspace file."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    with open(os.path.join(WORKSPACE_DIR, os.path.basename(filename)), 'w', encoding='utf-8') as f:
        f.write(content.strip())
    return f"Success: Wrote to '{filename}'."

def read_system_proc(query: str) -> str:
    """Read host OS /proc paths. Query must be 'cpu', 'mem', 'uptime', or 'version'."""
    path = os.path.join("/host_proc", query.lower() + ('info' if query.lower() in ['cpu', 'mem'] else ''))
    if not os.path.exists(path): return f"Path '{query}' missing."
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if query.lower() == 'mem':
            mem_total = mem_avail = 0
            for l in lines:
                if l.startswith('MemTotal:'): mem_total = int(l.split()[1])
                elif l.startswith('MemAvailable:'): mem_avail = int(l.split()[1])
            if mem_total and mem_avail:
                used = mem_total - mem_avail
                return f"Memory Used: {used/1024:.0f} MB / {mem_total/1024:.0f} MB ({(used/mem_total)*100:.1f}%)"
            return "".join([l for l in lines if 'Mem' in l or 'Swap' in l])
        if query.lower() == 'cpu': return f"{next((l for l in lines if 'model name' in l), 'Unknown')} | Cores: {len([l for l in lines if 'processor' in l])}"
        return "".join(lines)[:500]

def get_docker_info(query: str) -> str:
    """Query host Docker containers. Query must be 'list' or 'memory'."""
    try:
        containers = docker.from_env().containers.list()
        if not containers: return "No containers running."
        if query == 'list': return "\n".join([f"{c.name}: {c.status}" for c in containers])
        return "\n".join([f"{c.name}: {c.stats(stream=False).get('memory_stats', {}).get('usage', 0) / 1024**2:.2f} MB" for c in containers])
    except Exception as e: return f"Docker error: {e}"

def run_sandboxed_command(command: str) -> str:
    """Execute bash commands in a secure container."""
    try:
        out = docker.from_env().containers.run(
            "python:3.10-slim", command=["/bin/bash", "-c", f"timeout 30 {command}"], remove=True, mem_limit="512m",
            volumes={os.path.abspath(WORKSPACE_DIR): {'bind': '/workspace', 'mode': 'rw'}}, working_dir="/workspace"
        )
        res = out.decode().strip() or "Success (No output)."
        
        if len(res) > 500:
            return sub_agent_task(res, "Extract the specific error message or final outcome from this terminal log.")
        return res[:4000]
    except Exception as e: return f"Execution error: {e}"

def list_workspace_files(pattern: str = "*") -> str:
    """Search workspace file names. Pattern can be '*.txt' or '*'."""
    files = [os.path.relpath(f, WORKSPACE_DIR) for f in glob.glob(os.path.join(WORKSPACE_DIR, pattern), recursive=True) if os.path.isfile(f)]
    return "\n".join(files) if files else "No files found."

def fetch_url_content(url: str) -> str:
    """Extract raw text content from a web page URL."""
    try:
        soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).text, 'lxml')
        for script in soup(["script", "style", "nav", "footer"]): script.extract()
        return soup.get_text(separator=' ', strip=True)[:4000]
    except Exception as e: return f"Scraping error: {e}"

def scrape_and_summarize_url(url: str, goal: str) -> str:
    """Delegates webpage reading to a fast 0.5B sub-agent to summarize information based on a goal, then saves it to memory."""
    raw_text = fetch_url_content(url)
    if "error" in raw_text.lower(): return raw_text
    
    try:
        res = client.chat(model=SUB_MODEL_NAME, messages=[
            {"role": "system", "content": "You are a concise research sub-agent. Extract facts from the text that directly answer the Goal. No fluff."},
            {"role": "user", "content": f"Goal: {goal}\n\nText:\n{raw_text[:8000]}"}
        ], options={"num_predict": 400, "temperature": 0.1})
        summary = res.message.content.strip()
        
        if emb := get_ollama_embedding(summary):
            doc_id = hashlib.md5(summary.encode()).hexdigest()
            kb_collection.add(ids=[f"sub_{doc_id}"], embeddings=[emb], documents=[summary], metadatas=[{"source": url, "goal": goal}])
        return f"Sub-Agent Summary:\n{summary}\n[Saved to DB]"
    except Exception as e: return f"Sub-agent error: {e}"

def manage_processes(query: str) -> str:
    """Check running system processes. Query: 'count', 'top_cpu', 'top_mem', 'compositor', or a name."""
    try:
        procs = [p.info for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent'])]
        if query in ['count', 'all']: return f"Total active processes: {len(procs)}"
        if query == 'top_cpu': return "\n".join([f"{p['pid']}: {p['name']} ({p['cpu_percent']}%)" for p in sorted(procs, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5]])
        if query == 'top_mem': return "\n".join([f"{p['pid']}: {p['name']} ({p['memory_percent']:.1f}%)" for p in sorted(procs, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]])
        if query in ['compositor', 'wm', 'gui']:
            wms = {p['name'] for p in procs if p['name'] and any(w in p['name'].lower() for w in ['hyprland', 'wayland', 'xorg', 'sway', 'kwin', 'dwm'])}
            return f"Active WMs: {', '.join(wms)}" if wms else "No known WMs running."
        
        matches = [p for p in procs if p['name'] and query.lower() in p['name'].lower()]
        return "\n".join([f"PID {p['pid']}: {p['name']} (CPU: {p['cpu_percent'] or 0}%, RAM: {p['memory_percent'] or 0}%)" for p in matches]) if matches else "Process not found."
    except Exception as e: return f"Process error: {e}"

def query_mariadb(sql_query: str) -> str:
    """Execute read-only SELECT queries on the database."""
    if not all([DB_HOST, DB_USER, DB_PASS, DB_NAME]):
        return "Database not configured. Set DB_HOST, DB_USER, DB_PASS, and DB_NAME environment variables to enable this tool."
    if not sql_query.strip().upper().startswith("SELECT"): return "Error: Only SELECT permitted."
    try:
        with pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, cursorclass=pymysql.cursors.DictCursor) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchmany(10)
                raw_json = json.dumps(rows, indent=2) if rows else "0 rows returned."
                
                if rows:
                    return sub_agent_task(raw_json, "Translate this raw JSON SQL output into a clean, concise bulleted text summary.")
                return raw_json
    except Exception as e: return f"DB error: {e}"

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
    if kb_collection.count() == 0: return "Knowledge base empty."
    res = kb_collection.query(query_embeddings=[get_ollama_embedding(search_term)], n_results=3)
    return "\n\n---\n\n".join([f"[Source: {res['metadatas'][0][i].get('source', 'DB')}]\n{doc}" for i, doc in enumerate(res['documents'][0])]) if res['documents'][0] else "No info found."

def ingest_url_to_knowledge_base(url: str) -> str:
    """Permanently embed raw web URL text to vector DB."""
    if "error" in (text := fetch_url_content(url).lower()): return text
    chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
    count = 0
    for i, c in enumerate(chunks):
        doc_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
        if not kb_collection.get(ids=[doc_id])['ids']:
            kb_collection.add(ids=[doc_id], embeddings=[get_ollama_embedding(c)], documents=[c], metadatas=[{"source": url}])
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

# Dynamic bindings
AVAILABLE_TOOLS = [
    search_ddg, search_wikipedia, search_imdb, web_research, get_weather, read_file, write_file,
    read_system_proc, get_docker_info, run_sandboxed_command, list_workspace_files,
    fetch_url_content, scrape_and_summarize_url, manage_processes, query_mariadb,
    get_system_time, get_date_info, calculate_age, query_knowledge_base,
    ingest_url_to_knowledge_base, crawl_and_ingest_domain
]
AVAILABLE_FUNCTIONS = {f.__name__: f for f in AVAILABLE_TOOLS}

# --- Nomic Preprocessor Implementation ---
def get_ollama_embedding(text: str) -> list:
    try:
        return requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text}, timeout=10).json().get('embedding', [])
    except Exception: return []

def init_few_shot_db():
    if few_shot_collection.count() > 0: return
    print("Initializing Nomic Dynamic Preprocessor...")
    examples = [
        ("hello", "This is a casual greeting. Reply directly with a friendly hello. No tools needed."),
        ("hi there", "This is a casual greeting. Reply directly with a friendly hello. No tools needed."),
        ("What's the weather in London?", "Use get_weather."),
        ("What is the current wind speed?", "Use get_weather."),
        ("Who is Microsoft's CEO?", "Use search_ddg or search_wikipedia."),
        ("Age born on 1990-05-05?", "Use calculate_age and get_date_info with '1990-05-05'."),
        ("Ping google", "Use run_sandboxed_command with 'ping -c 4 google.com'."),
        ("Free RAM?", "Use read_system_proc with 'mem'."),
        ("Summarize this site", "You MUST use the scrape_and_summarize_url tool."),
        ("Search for recent news on X and summarize it", "Use web_research, not search_ddg alone."),
        ("What's out there about Y? Give me sources.", "Use web_research, not search_ddg alone."),
        ("What is the largest shark?", "DO NOT answer from memory. You MUST execute search_ddg or search_wikipedia first to verify."),
        ("How far away is the moon?", "DO NOT answer from memory. You MUST execute search_ddg or search_wikipedia first to verify."),
        ("Double check your answer", "Your previous answer is being challenged. DO NOT APOLOGIZE. You MUST execute the tool search_ddg immediately to find the truth."),
        ("Are you sure?", "Your previous answer is being challenged. DO NOT APOLOGIZE. You MUST execute the tool search_ddg immediately to find the truth.")
    ]
    for text, hint in examples:
        if emb := get_ollama_embedding(text):
            doc_id = hashlib.md5(text.encode()).hexdigest()
            few_shot_collection.add(ids=[doc_id], embeddings=[emb], documents=[hint])

def preprocess_user_prompt(user_input: str) -> str:
    # Clean the input to standard lowercase words
    lower_input = re.sub(r'[^a-z\s]', '', user_input.lower().strip())
    words = lower_input.split()
    
    # 1. Hardcoded Keyword Fallbacks 
    verification_triggers = [
        "are you sure", "double check", "verify that", "incorrect", 
        "wrong", "that is false", "not true", "bullshit", "untrue"
    ]
    if any(trigger in lower_input for trigger in verification_triggers):
        return f"{user_input}\n\n[SYSTEM HINT: Your previous answer is being challenged or corrected by the user. DO NOT APOLOGIZE or blindly agree. Do not assume you are wrong. You MUST immediately execute the search_ddg tool to pull real-time data and objectively verify the truth before replying.]"
        
    weather_triggers = ["weather", "temperature", "forecast", "wind", "speed", "conditions"]
    if any(trigger in lower_input for trigger in weather_triggers):
        return f"{user_input}\n\n[SYSTEM HINT: You MUST use the get_weather tool to find the current conditions.]"

    # URL and Web Scraping Interceptor
    url_triggers = ["http://", "https://", "www.", "scrape", "fetch", "summarize site"]
    if any(trigger in lower_input for trigger in url_triggers):
        return f"{user_input}\n\n[SYSTEM HINT: DO NOT state that you cannot browse the internet. You possess tools for this. You MUST execute the `fetch_url_content` or `scrape_and_summarize_url` tool to process the requested web page.]"

    # Time and Date Check
    time_triggers = ["what time", "current time", "date is it", "todays date", "what day"]
    if any(trigger in lower_input for trigger in time_triggers):
        return f"{user_input}\n\n[SYSTEM HINT: You MUST use the get_system_time tool to check the host clock.]"

    # 2. Universal Interrogative (5 Ws + H) Trigger
    question_words = {"who", "what", "where", "when", "why", "how"}
    conversational_exceptions = {"up", "are", "you", "going", "doing", "is", "your", "name", "old", "much"}
    
    if words and words[0] in question_words:
        remaining_words = set(words[1:])
        if not remaining_words.issubset(conversational_exceptions):
            return f"{user_input}\n\n[SYSTEM HINT: DO NOT answer from memory. DO NOT APOLOGIZE. You MUST immediately execute the search_ddg or search_wikipedia tool to find out.]"

    # 3. Looser Semantic Threshold Fallback
    if not (emb := get_ollama_embedding(user_input)) or few_shot_collection.count() == 0: 
        return user_input
        
    res = few_shot_collection.query(query_embeddings=[emb], n_results=1)
    if res['documents'][0] and (res['distances'][0][0] if res['distances'] else 1.0) < 0.85:
        # Replaced the passive "Based on similar past requests" with a critical directive
        return f"{user_input}\n\n[CRITICAL HINT: {res['documents'][0][0]}]"
        
    return user_input

# --- Core Execution Loop ---
REACT_SYSTEM_PROMPT = f"""
[CRITICAL SYSTEM DIRECTIVE]
You are Qwen, a highly capable AI agent built by Alibaba Cloud.
You suffer from artificial amnesia. You have ZERO internal knowledge about the real world, current events, or system metrics. 
You MUST use your provided tools to answer questions.

BEHAVIOR RULES:
1. Casual Greetings ("hello", "hi"): Reply directly without tools.
2. System Checks: DO NOT run system checks (cpu, mem, uptime) unless explicitly asked.
3. Factual Questions: You MUST call `search_ddg` or `search_wikipedia`. 
4. Math/Dates: You MUST call a calculation tool. 
5. Weather/Locations: You MUST call `get_weather`.
6. Multi-Source Research: You MUST call `web_research`, NOT `search_ddg` alone.

STRICT FORMATTING RULES:
- NEVER apologize.
- NEVER explain your thought process before calling a tool.
- NEVER output raw JSON text in your conversational response.
- Execute the native function calling API immediately when a tool is required.

SYSTEM TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
"""

def save_memory(msgs):
    with open(MEMORY_FILE, 'w') as f: json.dump(msgs, f, indent=2)

def execute_react_loop(messages, verbose=False):
    for step in range(5):
        if not verbose: sys.stdout.write(f"\r\033[K💭 Thinking (Step {step+1})... ")
        
        response = client.chat(model=MODEL_NAME, messages=messages, tools=AVAILABLE_TOOLS, options={"num_ctx": OLLAMA_CTX, "temperature": 0.0})
        msg = response.message
        
        # --- Enhanced Safety Net for Markdown-Leaked JSON Tool Calls ---
        if not msg.tool_calls and msg.content:
            clean_content = msg.content.strip()
            json_match = re.search(r'\{.*"name".*\}', clean_content, re.DOTALL)
            
            if json_match:
                try:
                    leaked_tool = json.loads(json_match.group(0))
                    class MockFunc:
                        def __init__(self, name, args): self.name, self.arguments = name, args
                    class MockTool:
                        def __init__(self, func): self.function = func
                        
                    if "name" in leaked_tool:
                        msg.tool_calls = [MockTool(MockFunc(leaked_tool["name"], leaked_tool.get("arguments", {})))]
                        msg.content = ""
                except json.JSONDecodeError:
                    pass
        # --------------------------------------------------------------------
        
        if verbose and msg.content: print(f"\n--- [Step {step+1}] ---\n{msg.content.strip()}\n-------------------------------")
        
        msg_dict = {"role": msg.role or "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [{"function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
            
        # SAFEGUARD: Do not append completely empty assistant turns to memory
        if msg_dict["content"].strip() or "tool_calls" in msg_dict:
            messages.append(msg_dict)
        else:
            # If the model bugged out and returned nothing, force a fallback to prevent history corruption
            fallback_text = "I encountered an empty response from the model. Please try again."
            messages.append({"role": "assistant", "content": fallback_text})
            msg.content = fallback_text  # <-- CRITICAL FIX: Ensures it prints!
        
        if not msg.tool_calls:
            if not verbose: sys.stdout.write("\r\033[K")
            return msg.content or "", messages
            
        for tool in msg.tool_calls:
            func_name = tool.function.name
            raw_args = tool.function.arguments
            
            func_args = {}
            if raw_args:
                for k, v in raw_args.items():
                    if isinstance(v, dict) and 'value' in v:
                        func_args[k] = v['value']
                    else:
                        func_args[k] = v
            
            if verbose: print(f"🛠️  Executing: {func_name}({func_args})")
            else: sys.stdout.write(f"\r\033[K🛠️  Running tool: {func_name}... ")
            
            try:
                obs = str(AVAILABLE_FUNCTIONS[func_name](**func_args))[:1500] if func_name in AVAILABLE_FUNCTIONS else f"Error: Tool '{func_name}' not found."
            except Exception as e:
                obs = f"Execution error in {func_name}: {e}"
                
            if verbose: print(f"👁️  Observation loaded.\n")
            messages.append({"role": "tool", "name": func_name, "content": obs})
            
    if not verbose: sys.stdout.write("\r\033[K")
    return "Error: Max iterations reached.", messages

def main():
    print("Connecting & Pulling Models...", flush=True)
    
    for i in range(30):  # Increased to 30 for a 60-second wait window
        try:
            client.pull(MODEL_NAME)
            client.pull(SUB_MODEL_NAME)
            break
        except Exception:
            if i == 29: sys.exit("Ollama connection error: Server did not respond.")
            time.sleep(2)

    global kb_collection, few_shot_collection
    
    init_few_shot_db()
    messages = json.load(open(MEMORY_FILE)) if os.path.exists(MEMORY_FILE) else [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    verbose_mode = False
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== Agent Active ({MODEL_NAME}) ===\nTools loaded: {len(AVAILABLE_TOOLS)}\nCommands: /think, /wipe, /wipe_kb, /copy, exit\n")

    session = PromptSession(history=FileHistory(CMD_HISTORY_FILE))
    completer = WordCompleter(['/think', '/wipe', '/wipe_kb', '/copy', 'exit', 'quit'] + list(AVAILABLE_FUNCTIONS.keys()), ignore_case=True)
    last_response = ""

    while True:
        try:
            user_input = session.prompt("👤: ", completer=completer).strip()
            if not user_input: continue
            
            if (cmd := user_input.lower()) in ['exit', 'quit']:
                save_memory(messages); break
            if cmd == '/think':
                verbose_mode = not verbose_mode; print(f"🔧 Verbose mode: {'ON' if verbose_mode else 'OFF'}"); continue
            if cmd == '/wipe':
                messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]; save_memory(messages); print("🧹 Memory wiped."); continue
            
            if cmd == '/wipe_kb':
                for coll_name in ["agent_knowledge", "agent_few_shot"]:
                    try: chroma_client.delete_collection(coll_name)
                    except Exception: pass
                
                try:
                    kb_collection = chroma_client.get_or_create_collection("agent_knowledge")
                    few_shot_collection = chroma_client.get_or_create_collection("agent_few_shot")
                    init_few_shot_db()
                    print("🧠 KB and Few-Shot databases wiped and reset.")
                except Exception as e: 
                    print(f"⚠️ Error recreating collections: {e}")
                continue
            
            if cmd == '/copy':
                copy_to_clipboard(last_response); print("📋 Copied!"); continue
            
            enriched_input = preprocess_user_prompt(user_input)
            if verbose_mode and enriched_input != user_input: print("✨ [Nomic Preprocessor Triggered]")

            messages.append({"role": "user", "content": enriched_input})
            last_response, messages = execute_react_loop(messages, verbose=verbose_mode)
            print(f"💻: {last_response}")
            save_memory(messages)

        except (KeyboardInterrupt, EOFError):
            save_memory(messages); break

if __name__ == "__main__":
    main()
