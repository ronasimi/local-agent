import os
import json
import sys
import re
import requests
import base64
from datetime import datetime
from ollama import Client
from ddgs import DDGS
import wikipedia
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "qwen2.5"
MEMORY_FILE = "/app/memory/chat_history.json"
CMD_HISTORY_FILE = "/app/memory/cmd_history.txt"
WORKSPACE_DIR = "/app/workspace"

client = Client(host=OLLAMA_HOST)

def copy_to_clipboard(text: str):
    """Uses ANSI OSC 52 escape sequence to copy text to the host clipboard from inside Docker."""
    encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    sys.stdout.write(f"\033]52;c;{encoded}\a")
    sys.stdout.flush()

# Tool Definitions
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
    """Search IMDb for movie, TV show, or actor information using a targeted web search."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(f"site:imdb.com {query}", max_results=3)]
            if not results:
                return f"No IMDb results found for '{query}'."
            return "\n\n".join([f"IMDb Title: {r['title']}\nDetails: {r['body']}" for r in results])
    except Exception as e:
        return f"IMDb search error: {str(e)}"

def get_weather(location: str) -> str:
    """Get the current live weather conditions for a specific city or location."""
    try:
        custom_format = "%l:+%C+(%c),+%t"
        response = requests.get(f"https://wttr.in/{location}?format={custom_format}", timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        return f"Weather service returned status code {response.status_code}."
    except Exception as e:
        return f"Weather API error: {str(e)}"

def read_file(filename: str) -> str:
    """Reads the contents of a local file in the workspace."""
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
    """Writes content to a local file in the workspace."""
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

def read_system_proc(filepath: str) -> str:
    """Reads read-only system information from the host /proc directory."""
    HOST_PROC_DIR = "/host_proc"
    clean_path = filepath.lstrip('/')
    full_path = os.path.abspath(os.path.join(HOST_PROC_DIR, clean_path))
    
    if not full_path.startswith(HOST_PROC_DIR):
        return "Error: Path traversal attempt blocked."
        
    if not os.path.exists(full_path):
        return f"Error: System path '{clean_path}' does not exist."
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()[:2000]
    except Exception as e:
        return f"Error reading system file: {str(e)}"

def get_system_time(query: str = "") -> str:
    """Returns the current date and time of the host system."""
    return f"The current system date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

tools_map = {
    'search_ddg': search_ddg,
    'search_wikipedia': search_wikipedia,
    'search_imdb': search_imdb,
    'get_weather': get_weather,
    'read_file': read_file,
    'write_file': write_file,
    'read_system_proc': read_system_proc,
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
- search_ddg: Search the live internet for recent news, events, or general web searches. 🌐
- search_wikipedia: Search Wikipedia for deep background on concepts or history. 📚
- search_imdb: Search IMDb for movie, TV show, or actor details (ratings, cast, plot, etc.). 🎬
- get_weather: Use this explicitly to find current weather conditions or temperatures for a specific location. 🌤️
- read_file: Read the text contents of a file in your workspace. 📄
- write_file: Write text to a file in your workspace. The Action Input MUST be formatted exactly as 'filename.txt|Your content here'. ✍️
- read_system_proc: Read hardware and system info from the host OS. YOU MUST USE THIS TOOL if the user asks about "your" memory, RAM, CPU, or system specs. Common inputs: 'cpuinfo', 'meminfo', 'uptime', 'version'. 💻
- get_system_time: Use this to check the current date, time, or year. 🕒

⚙️  OUTPUT FORMAT (TWO PATHS):

PATH 1: CASUAL CHAT (FAST-TRACK)
If the user is just saying hello, thanking you, or asking a casual question that requires NO tools, completely skip the Thought and Action steps. 
Immediately output:
Final Answer: your conversational response.
⚠️ CRITICAL: DO NOT use this path if the user asks for LIVE or FACTUAL data (like weather, system stats, current time, news, or search). You have no internal knowledge of the present. For those, you MUST use PATH 2.

PATH 2: TOOL USE REQUIRED
If you need to look up information, read/write files, or check the system, you MUST use the strict ReAct format:
Thought: your internal reasoning about what to do next
Action: the exact tool name (e.g., read_system_proc)
Action Input: the query to pass to the tool

(You will then receive an Observation from the system, and you can repeat this cycle 🔄)
When you have the information, conclude with:
Thought: I now have the final answer. 💡
Final Answer: your response to the user.

🚨 CRITICAL RULES:
- NEVER generate the "Observation:" text yourself! 🛑
- NEVER output a "Final Answer" in the same step as an "Action". After writing the Action Input, you must STOP! ⚠️
- NEVER use placeholders like [Insert Data Here]. If you do not know the information, you MUST use a tool to find it before taking any other action.
- TOOL CHAINING: If you are asked to save live data to a file, you must first use a tool (like get_weather) to gather the data, wait for the Observation, and THEN use the write_file tool in your next step.
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
    """Extracts the action and input, or the final answer from the LLM's text."""
    action_match = re.search(r"Action:\s*(.*)", text)
    input_match = re.search(r"Action Input:\s*(.*)", text)
    
    # 1. Did it successfully call a tool?
    if action_match and input_match:
        return {
            "type": "action",
            "action": action_match.group(1).strip(),
            "input": input_match.group(1).strip()
        }
        
    # 2. Did it explicitly declare a final answer?
    if "Final Answer:" in text:
        return {"type": "finish", "content": text.split("Final Answer:")[-1].strip()}
        
    # 3. Fast-Track Conversational Fallback
    # If it attempted to use a tool but broke syntax, throw an error.
    if "Action:" in text or "Action Input:" in text:
        return {"type": "invalid", "content": text.strip()}
        
    # If no tool words are present, assume it's a fast-tracked conversational response.
    # We strip out stray "Thought:" tags just in case it thought for a second before answering.
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
                "num_ctx": 8192,
                "stop": [
                    "Observation:", 
                    "\nObservation:"
                ]
            }
        )
        
        content = response.message.content
        
        if verbose:
            print(f"\n--- [Internal Monologue Step {step+1}] ---\n{content.strip()}\n-------------------------------")
        
        messages.append({"role": "assistant", "content": content})
        
        parsed = parse_react_output(content)
        
        if parsed["type"] == "finish":
            if not verbose:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            return parsed["content"], messages
            
        elif parsed["type"] == "invalid":
            feedback = "System Notice: You provided a Thought but did not specify an Action or Final Answer. You must pick an explicit Action and Action Input from the available tools list to proceed."
            if verbose:
                print(f"⚠️  Format Error caught. Sending correction feedback to model.")
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
                
            observation = observation[:1500] 
            
            if verbose:
                print(f"👁️  System Observation: {observation[:75]}...\n")
            
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

    print(f"\n🌟 === ReAct Agent Active (Model: {MODEL_NAME}) === 🌟\n")
    print("🛠️  AVAILABLE TOOLS:")
    print("  • search_ddg       : Search the live internet for recent news/events 🌐")
    print("  • search_wikipedia : Search Wikipedia for deep background info 📚")
    print("  • search_imdb      : Search IMDb for movie, TV show, or actor details 🎬")
    print("  • get_weather      : Find current weather conditions for a location 🌤️")
    print("  • read_file        : Read the text contents of a workspace file 📄")
    print("  • write_file       : Write text to a file in your workspace ✍️")
    print("  • read_system_proc : Read hardware and system info from the host OS 💻")
    print("  • get_system_time  : Check the current date, time, or year 🕒\n")
    print("⌨️  COMMANDS:")
    print("  • /think           : Toggle verbose internal monologue 🧠")
    print("  • /wipe            : Clear chat memory and start fresh 🧹")
    print("  • /copy            : Copy the last agent response to clipboard 📋")
    print("  • exit / quit      : End the session and save history 🛑")
    print("=========================================================\n")

    bad_file = os.path.join(WORKSPACE_DIR, "filename=tokyo_weather.txt")
    if os.path.exists(bad_file):
        try: os.remove(bad_file)
        except Exception: pass

    os.makedirs(os.path.dirname(CMD_HISTORY_FILE), exist_ok=True)
    
    autocomplete_words = ['/think', '/wipe', '/copy', 'exit', 'quit'] + list(tools_map.keys())
    completer = WordCompleter(autocomplete_words, ignore_case=True)
    
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
                
            if not user_input:
                continue
                
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

            if user_input.lower() == '/copy':
                if last_agent_response:
                    copy_to_clipboard(last_agent_response)
                    print("📋 Copied last response to host clipboard!")
                else:
                    print("⚠️ Nothing to copy yet.")
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
