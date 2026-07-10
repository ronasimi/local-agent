import os
import json
import sys
import re
import time
import requests
import base64
import docker
import glob
import psutil
import pymysql
import chromadb
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
from ollama import Client
from ddgs import DDGS
import wikipedia
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "qwen2.5-coder:3b"

# Memory Paths
MEMORY_DIR = os.getenv("MEMORY_DIR", "/app/memory")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/app/workspace")
MEMORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json")
CMD_HISTORY_FILE = os.path.join(MEMORY_DIR, "cmd_history.txt")

client = Client(host=OLLAMA_HOST)

# --- Knowledge Base Setup ---
CHROMA_PATH = os.path.join(MEMORY_DIR, "vector_db")
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
kb_collection = chroma_client.get_or_create_collection(name="agent_knowledge")

def copy_to_clipboard(text: str):
    """Uses ANSI OSC 52 escape sequence to copy text to the host clipboard from inside Docker."""
    encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    sys.stdout.write(f"\033]52;c;{encoded}\a")
    sys.stdout.flush()

# --- Core Tools ---
def search_ddg(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No web results found."
            return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"DuckDuckGo search error: {str(e)}"

def search_wikipedia(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

def search_imdb(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(f"site:imdb.com {query}", max_results=3)]
            if not results:
                return f"No IMDb results found for '{query}'."
            return "\n\n".join([f"IMDb Title: {r['title']}\nDetails: {r['body']}" for r in results])
    except Exception as e:
        return f"IMDb search error: {str(e)}"

def get_weather(location: str) -> str:
    try:
        custom_format = "%l:+%C+(%c),+%t"
        response = requests.get(f"https://wttr.in/{location}?format={custom_format}", timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        return f"Weather service returned status code {response.status_code}."
    except Exception as e:
        return f"Weather API error: {str(e)}"

def read_file(filename: str) -> str:
    safe_filename = filename.replace("filename=", "").strip()
    safe_filename = os.path.basename(safe_filename)
    filepath = os.path.join(WORKSPACE_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        return f"Error: File '{safe_filename}' does not exist."
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(filename_and_content: str) -> str:
    try:
        filename, content = filename_and_content.split('|', 1)
        safe_filename = filename.replace("filename=", "").strip()
        safe_filename = os.path.basename(safe_filename)
        filepath = os.path.join(WORKSPACE_DIR, safe_filename)
        
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        return f"Success: Wrote to '{safe_filename}'."
    except ValueError:
        return "Error: Action Input must be formatted as 'filename.txt|content'"
    except Exception as e:
        return f"Error writing file: {str(e)}"

# --- System & Execution Tools ---
def parse_system_data(key: str, raw_data: str) -> str:
    """Cleans and formats raw /proc data for better LLM consumption."""
    lines = raw_data.split('\n')
    if key == 'meminfo':
        return "\n".join([line for line in lines if 'Mem' in line or 'Swap' in line])
    if key == 'cpuinfo':
        model = next((l for l in lines if 'model name' in l), "Unknown CPU")
        cores = len([l for l in lines if 'processor' in l])
        return f"{model.strip()} | Cores: {cores}"
    
    # --- OBSERVATION INJECTION ---
    if key == 'version':
        base_output = raw_data[:500]
        hint = "\n[System Hint: This only shows the kernel. To find the active compositor or window manager, use manage_processes.]"
        return base_output + hint
    # -----------------------------
        
    return raw_data[:500]

def read_system_proc(query: str) -> str:
    HOST_PROC_DIR = "/host_proc"
    proc_map = {'cpu': 'cpuinfo', 'mem': 'meminfo', 'uptime': 'uptime', 'version': 'version'}
    filename = proc_map.get(query.lower(), query.lower())
    full_path = os.path.join(HOST_PROC_DIR, filename)
    
    if not os.path.exists(full_path):
        return f"Error: System path '{filename}' does not exist."
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            raw = f.read()
            return parse_system_data(filename, raw)
    except Exception as e:
        return f"Error reading system file: {str(e)}"

def get_docker_info(query: str) -> str:
    try:
        docker_client = docker.from_env()
        containers = docker_client.containers.list()
        
        if not containers:
            return "No running containers found."

        query = query.lower()
        if "list" in query or "status" in query:
            return "\n".join([f"{c.name}: {c.status}" for c in containers])
            
        elif "memory" in query or "stats" in query:
            results = []
            for c in containers:
                stats = c.stats(stream=False)
                mem_stats = stats.get('memory_stats', {})
                mem_usage = mem_stats.get('usage', 0)
                if mem_usage > 0:
                    mem_mb = mem_usage / (1024 * 1024)
                    results.append(f"{c.name}: {mem_mb:.2f} MB")
                else:
                    results.append(f"{c.name}: Memory data unavailable")
            return "\n".join(results)
        return "Command not recognized. Try 'list', 'status', or 'memory'."
    except Exception as e:
        return f"Docker API error: {str(e)}"

def run_sandboxed_command(command: str) -> str:
    """Executes a shell command inside a temporary, isolated Docker container."""
    try:
        docker_client = docker.from_env()
        
        # We mount the workspace so the sandbox can interact with scripts created by the agent
        # Use absolute path for the host side, assuming WORKSPACE_DIR is mapped appropriately or just bind the container path
        output = docker_client.containers.run(
            image="python:3.10-slim",
            command=["/bin/sh", "-c", command],
            remove=True,                  
            mem_limit="512m",             
            cpu_period=100000,
            cpu_quota=50000,              
            network_mode="bridge",        
            volumes={os.path.abspath(WORKSPACE_DIR): {'bind': '/workspace', 'mode': 'rw'}},
            working_dir="/workspace"      
        )
        
        result = output.decode('utf-8').strip()
        if not result:
            return "Command executed successfully, but returned no output."
            
        return result[:4000] + "\n...[TRUNCATED]" if len(result) > 4000 else result
        
    except docker.errors.ContainerError as e:
        error_output = e.stderr.decode('utf-8').strip() if e.stderr else str(e)
        return f"Command failed with exit code {e.exit_status}:\n{error_output}"
    except Exception as e:
        return f"Sandbox execution error: {str(e)}"

def list_workspace_files(query: str = "") -> str:
    try:
        search_pattern = query if query and query.lower() != 'all' else '**/*'
        full_pattern = os.path.join(WORKSPACE_DIR, search_pattern)
        files = glob.glob(full_pattern, recursive=True)
        files = [os.path.relpath(f, WORKSPACE_DIR) for f in files if os.path.isfile(f)]
        if not files: return f"No files found matching '{query}' in workspace."
        return "\n".join(files)
    except Exception as e:
        return f"File listing error: {str(e)}"

def fetch_url_content(url: str) -> str:
    if not url.startswith('http'): return "Error: URL must start with http:// or https://"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        text = soup.get_text(separator=' ', strip=True)
        return text[:4000] + "\n...[TRUNCATED]" if len(text) > 4000 else text
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def manage_processes(query: str) -> str:
    """Checks running processes and process counts."""
    try:
        query = query.strip().lower()
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            processes.append(proc.info)

        if query in ['list', 'all', 'count', 'status']:
            return f"Total active processes: {len(processes)}\n(Note: To see specific processes, query 'top_cpu', 'top_mem', or a specific process name)."

        if query == 'top_cpu':
            top = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)[:5]
            return "\n".join([f"PID {p['pid']}: {p['name']} (CPU: {p['cpu_percent']}%)" for p in top])
            
        elif query == 'top_mem':
            top = sorted(processes, key=lambda p: p['memory_percent'], reverse=True)[:5]
            return "\n".join([f"PID {p['pid']}: {p['name']} (RAM: {p['memory_percent']:.1f}%)" for p in top])
            
        # --- NEW: Categorical search for compositors ---
        elif query in ['compositor', 'wm', 'desktop', 'gui']:
            wm_list = ['hyprland', 'wayland', 'xorg', 'xwayland', 'sway', 'kwin', 'mutter', 'gnome-shell', 'xfwm4', 'dwm', 'i3', 'openbox']
            matches = [p for p in processes if p['name'] and any(wm in p['name'].lower() for wm in wm_list)]
            if not matches:
                return "No known compositor/window manager processes found running."
            
            # Extract unique names to prevent spamming if there are 20 hyprland threads
            unique_wms = set([p['name'] for p in matches])
            return "Active compositor/WM processes found: " + ", ".join(unique_wms)
        # -----------------------------------------------

        else:
            matches = [p for p in processes if p['name'] and query in p['name'].lower()]
            if not matches: 
                return f"No processes found matching '{query}'. Valid inputs are 'count', 'top_cpu', 'top_mem', 'compositor', or a specific process name."
            return "\n".join([f"PID {p['pid']}: {p['name']} (CPU: {p['cpu_percent']}%, RAM: {p['memory_percent']:.1f}%)" for p in matches])
            
    except Exception as e:
        return f"Process manager error: {str(e)}"

def query_mariadb(sql_query: str) -> str:
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: For safety, this tool only permits SELECT queries."
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', '192.168.1.100'),
            user=os.getenv('DB_USER', 'kodi'),
            password=os.getenv('DB_PASS', 'kodi'),
            database=os.getenv('DB_NAME', 'kodi_video121'),
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchmany(10)
        if not result: return "Query successful, but returned 0 rows."
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Database error: {str(e)}"
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()

def get_system_time(query: str = "") -> str:
    return f"The current system date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

# --- Knowledge Base Tools ---
def get_ollama_embedding(text: str) -> list:
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": "nomic-embed-text", "prompt": text}
    response = requests.post(url, json=payload).json()
    return response.get('embedding', [])

def ingest_url_to_knowledge_base(url: str) -> str:
    """Scrapes a single URL and saves its chunks to ChromaDB."""
    try:
        text_content = fetch_url_content(url)
        if "Error" in text_content: return text_content
            
        chunks = [c.strip() for c in text_content.split('\n\n') if len(c.strip()) > 50]
        ingested = 0
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{url}_chunk_{i}"
            if kb_collection.get(ids=[doc_id])['ids']: continue
                
            embedding = get_ollama_embedding(chunk)
            if embedding:
                kb_collection.add(
                    ids=[doc_id], embeddings=[embedding],
                    documents=[chunk], metadatas=[{"source": url}]
                )
                ingested += 1
                
        return f"Successfully ingested {ingested} chunks from {url} into the knowledge base."
    except Exception as e:
        return f"Knowledge base ingestion error: {str(e)}"

def crawl_and_ingest_domain(start_url_and_max: str) -> str:
    """Recursively crawls a domain and ingests pages into the knowledge base."""
    try:
        parts = start_url_and_max.split('|')
        start_url = parts[0].strip()
        max_pages = int(parts[1].strip()) if len(parts) > 1 else 3
    except ValueError:
        return "Error: Action Input must be formatted exactly as 'https://example.com|5'"

    if not start_url.startswith('http'):
        return "Error: URL must start with http:// or https://"

    base_domain = urlparse(start_url).netloc
    visited, queue = set(), [start_url]
    ingested_count = 0
    log = []

    while queue and ingested_count < max_pages:
        current_url = queue.pop(0)
        if current_url in visited: continue
            
        visited.add(current_url)
        log.append(f"Crawling: {current_url}")
        
        # Ingest
        ingest_result = ingest_url_to_knowledge_base(current_url)
        if "Successfully" in ingest_result:
            ingested_count += 1
            
        # Discover Links
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(current_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'lxml')
            
            for link in soup.find_all('a', href=True):
                next_url = urljoin(current_url, link['href']).split('#')[0]
                if urlparse(next_url).netloc == base_domain and next_url not in visited and next_url not in queue:
                    if not any(next_url.lower().endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.zip']):
                        queue.append(next_url)
        except Exception as e:
            log.append(f"  -> Failed to parse links from {current_url}: {str(e)}")
            
        time.sleep(1) # Be polite to the server

    log.append(f"Crawl complete. Successfully ingested {ingested_count} pages.")
    return "\n".join(log)

def query_knowledge_base(search_term: str) -> str:
    """Searches the agent's internal vector database."""
    try:
        if kb_collection.count() == 0: return "The knowledge base is empty. Ingest URLs first."
            
        query_embedding = get_ollama_embedding(search_term)
        results = kb_collection.query(query_embeddings=[query_embedding], n_results=3)
        
        if not results['documents'][0]: return "No relevant information found."
            
        output = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            output.append(f"[Source: {source}]\n{doc}")
            
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Knowledge base search error: {str(e)}"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Tool Mapping & Prompts ---
tools_map = {
    'search_ddg': search_ddg,
    'search_wikipedia': search_wikipedia,
    'search_imdb': search_imdb,
    'get_weather': get_weather,
    'read_file': read_file,
    'write_file': write_file,
    'read_system_proc': read_system_proc,
    'get_docker_info': get_docker_info,
    'run_sandboxed_command': run_sandboxed_command,
    'list_workspace_files': list_workspace_files,
    'fetch_url_content': fetch_url_content,
    'manage_processes': manage_processes,
    'query_mariadb': query_mariadb,
    'ingest_url_to_knowledge_base': ingest_url_to_knowledge_base,
    'crawl_and_ingest_domain': crawl_and_ingest_domain,
    'query_knowledge_base': query_knowledge_base,
    'get_system_time': get_system_time
}

current_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

REACT_SYSTEM_PROMPT = f"""
🌟 ========================================= 🌟
     🤖 AUTONOMOUS REACT AGENT PROTOCOL 🤖
🌟 ========================================= 🌟

You are an autonomous AI agent operating on the ReAct (Reason + Act) framework. 🧠✨
SYSTEM AWARENESS: The current date and time is {current_time}.

🛠️  AVAILABLE TOOLS:
- search_ddg: Search the live internet for recent news/events. 🌐
- search_wikipedia: Search Wikipedia for deep background on concepts. 📚
- search_imdb: Search IMDb for movie, TV show, or actor details. 🎬
- get_weather: Find current weather conditions for a specific location. 🌤️
- read_file: Read the text contents of a file in your workspace. 📄
- write_file: Write text to a file in your workspace. Action Input MUST be 'filename.txt|Your content'. ✍️
- read_system_proc: Read host OS system info. Inputs: 'cpu', 'mem', 'uptime', 'version'. 💻
- get_docker_info: Query the host Docker daemon. Inputs: 'list', 'status', or 'memory'. 🐳
- run_sandboxed_command: Execute bash commands or python scripts in a secure, isolated Linux sandbox. The sandbox has read/write access to your workspace folder. Action input MUST be the raw shell command (e.g., 'python script.py' or 'ping -c 4 google.com'). 🛡️
- list_workspace_files: List files matching a pattern in the workspace ('*.txt' or 'all'). 📁
- fetch_url_content: Fetch and read text from a specific webpage URL. 🔗
- manage_processes: Check running system processes. Inputs: 'count', 'top_cpu', 'top_mem', 'compositor', or a process name. ⚙️
- query_mariadb: Execute a read-only SELECT query on the local MariaDB database. 🗄️
- ingest_url_to_knowledge_base: Read a single webpage and permanently save it to your vector memory. Action input MUST be a valid URL. 📥
- crawl_and_ingest_domain: Recursively spider a website to build your knowledge base. Action input MUST be formatted as 'url|max_pages' (e.g., 'https://wiki.archlinux.org/|5'). 🕷️
- query_knowledge_base: Search your permanent vector memory for facts or previously ingested data. Action input should be a precise search term. 🧠
- get_system_time: Check the current date, time, or year. 🕒

⚙️  OUTPUT FORMAT (TWO PATHS):

PATH 1: CASUAL CHAT (FAST-TRACK)
If the user is just saying hello, thanking you, or asking a casual question that requires NO tools, completely skip the Thought and Action steps. 
Immediately output:
Final Answer: your conversational response.
⚠️ CRITICAL: DO NOT use this path if the user asks for LIVE or FACTUAL data. For those, you MUST use PATH 2.

PATH 2: TOOL USE REQUIRED
If you need to look up information, read files, or check the system, you MUST use the strict ReAct format:
Thought: your internal reasoning about what to do next
Action: the exact tool name 
Action Input: the query to pass to the tool

(You will then receive an Observation from the system, and you can repeat this cycle 🔄)
When you have the information, conclude with:
Thought: I now have the final answer. 💡
Final Answer: your response to the user.

🚨 CRITICAL RULES:
- NEVER generate the "Observation:" text yourself! 🛑
- NEVER output a "Final Answer" in the same step as an "Action". After writing the Action Input, you must STOP! ⚠️
- NEVER use placeholders like [Insert Data Here]. 
"""

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(messages):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(messages, f, indent=2)

def parse_react_output(text):
    action_match = re.search(r"Action:\s*(.*)", text)
    input_match = re.search(r"Action Input:\s*(.*)", text)
    
    if action_match and input_match:
        return {
            "type": "action",
            "action": action_match.group(1).strip(),
            "input": input_match.group(1).strip()
        }
        
    if "Final Answer:" in text:
        return {"type": "finish", "content": text.split("Final Answer:")[-1].strip()}
        
    if "Action:" in text or "Action Input:" in text:
        return {"type": "invalid", "content": text.strip()}
        
    clean_text = text.replace("Thought:", "").strip()
    return {"type": "finish", "content": clean_text}

def execute_react_loop(messages, verbose=False):
    max_steps = 5
    
    for step in range(max_steps):
        if not verbose:
            sys.stdout.write(f"\r\033[K💭 Thinking (Step {step+1})... ")
            sys.stdout.flush()

        response = client.chat(
            model=MODEL_NAME, 
            messages=messages,
            options={
                "num_ctx": 16384,
                "stop": ["Observation:", "\nObservation:"]
            }
        )
        
        try:
            metrics = dict(response)
            eval_count = metrics.get('eval_count', 0)
            eval_duration = metrics.get('eval_duration', 0)
            if eval_count and eval_duration:
                tps = eval_count / (eval_duration / 1e9)
                tps_string = f" ⚡ [{tps:.1f} t/s]"
            else:
                tps_string = ""
        except Exception:
            tps_string = ""
        
        content = response.message.content
        
        if verbose:
            print(f"\n--- [Internal Monologue Step {step+1}] ---\n{content.strip()}\n{tps_string}\n-------------------------------")
        
        messages.append({"role": "assistant", "content": content})
        parsed = parse_react_output(content)
        
        if parsed["type"] == "finish":
            if not verbose:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            return parsed["content"] + tps_string, messages
            
        elif parsed["type"] == "invalid":
            feedback = "System Notice: You provided a Thought but did not specify an Action or Final Answer. You must pick an explicit Action and Action Input from the available tools list to proceed."
            if verbose: print(f"⚠️  Format Error caught. Sending correction feedback.")
            messages.append({"role": "user", "content": feedback})
            continue
            
        elif parsed["type"] == "action":
            tool_name = parsed["action"]
            tool_input = parsed["input"]
            
            if verbose:
                print(f"🛠️  System Executing: {tool_name}('{tool_input}')")
            else:
                sys.stdout.write(f"\r\033[K🛠️  Running tool: {tool_name}... ")
                sys.stdout.flush()
            
            if tool_name in tools_map:
                observation = tools_map[tool_name](tool_input)
            else:
                observation = f"Error: Tool '{tool_name}' not recognized."
                
            observation = str(observation)[:1500] 
            
            if verbose: print(f"👁️  System Observation: {observation[:75]}...\n")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            
    if not verbose:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
    return "Error: Reached maximum iterations without a final answer.", messages

def main():
    print("Connecting to Ollama service...")
    try:
        client.pull(MODEL_NAME)
    except Exception as e:
        print(f"Error connecting/pulling model: {e}")
        sys.exit(1)

    messages = load_memory()
    verbose_mode = False
    
    if not messages:
        messages.append({"role": "system", "content": REACT_SYSTEM_PROMPT})
    clear_screen()
    print(f"\n=== ReAct Agent Active (Model: {MODEL_NAME}) ===\n")
    print("🛠️  AVAILABLE TOOLS:")
    print("  🌐 search_ddg          : Search the live internet for recent news/events")
    print("  📚 search_wikipedia    : Search Wikipedia for deep background info")
    print("  🌤️ get_weather         : Find current weather conditions for a location")
    print("  📄 read_file           : Read the text contents of a workspace file")
    print("  ✍️ write_file          : Write text to a file in your workspace")
    print("  💻 read_system_proc    : Read host OS system info (CPU, RAM, Uptime)")
    print("  🐳 get_docker_info     : Query running Docker containers (list, status, memory)")
    print("  🛡️ run_sandboxed_cmd   : Run scripts/commands in a secure isolated sandbox")
    print("  📁 list_workspace_files: List files matching a pattern in the workspace")
    print("  🔗 fetch_url_content   : Fetch and extract text from a webpage")
    print("  ⚙️ manage_processes    : Check running processes (count, top_cpu, top_mem, search)")
    print("  🗄️ query_mariadb       : Execute read-only queries on the MariaDB database")
    print("  📥 ingest_url          : Save a single URL to your vector memory")
    print("  🕷️ crawl_domain        : Spider and ingest an entire domain (max depth)")
    print("  🧠 query_kb            : Search your permanent local vector database\n")
    print("⌨️  COMMANDS:")
    print("  🧠 /think              : Toggle verbose internal monologue")
    print("  🧹 /wipe               : Clear chat memory and start fresh")
    print("  💥 /wipe_kb            : Delete and reset the ChromaDB vector database")
    print("  📋 /copy               : Copy the last agent response to clipboard")
    print("  🛑 exit / quit         : End the session and save history")
    print("=========================================================\n")

    bad_file = os.path.join(WORKSPACE_DIR, "filename=tokyo_weather.txt")
    if os.path.exists(bad_file):
        try: os.remove(bad_file)
        except Exception: pass

    os.makedirs(os.path.dirname(CMD_HISTORY_FILE), exist_ok=True)
    autocomplete_words = ['/think', '/wipe', '/wipe_kb', '/copy', 'exit', 'quit'] + list(tools_map.keys())
    completer = WordCompleter(autocomplete_words, ignore_case=True)
    
    global kb_collection 
    session = PromptSession(history=FileHistory(CMD_HISTORY_FILE))

    first_prompt = True
    last_agent_response = ""

    while True:
        try:
            if first_prompt:
                user_input = session.prompt("👤: ", completer=completer).strip()
                first_prompt = False
            else:
                user_input = session.prompt("\n👤: ", completer=completer).strip()
                
            if not user_input: continue
                
            if user_input.lower() in ['exit', 'quit']:
                save_memory(messages)
                print("History saved. Goodbye!")
                break
                
            if user_input.lower() == '/think':
                verbose_mode = not verbose_mode
                status = "ON" if verbose_mode else "OFF"
                print(f"🔧 Verbose thinking mode turned {status}.")
                continue
                
            if user_input.lower() == '/wipe':
                messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
                save_memory(messages)
                print("🧹 Memory wiped. Context reset!")
                continue

            if user_input.lower() == '/wipe_kb':
                try:
                    chroma_client.delete_collection(name="agent_knowledge")
                    kb_collection = chroma_client.create_collection(name="agent_knowledge")
                    print("🧠 Knowledge base wiped completely!")
                except Exception as e:
                    print(f"⚠️ Error wiping knowledge base: {e}")
                continue

            if user_input.lower() == '/copy':
                if last_agent_response:
                    copy_to_clipboard(last_agent_response)
                    print("📋 Copied last response to host clipboard!")
                else: print("⚠️ Nothing to copy yet.")
                continue
            
            messages.append({"role": "user", "content": user_input})
            final_response, messages = execute_react_loop(messages, verbose=verbose_mode)
            last_agent_response = final_response
            print(f"💻: {final_response}")
            save_memory(messages)

        except (KeyboardInterrupt, EOFError):
            save_memory(messages)
            print("\nSession saved. Exiting.")
            break

if __name__ == "__main__":
    main()
