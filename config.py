import os
import requests
from ollama import Client
import chromadb

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_CTX = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"))
MODEL_NAME = "qwen2.5-coder:3b"
SUB_MODEL_NAME = "qwen2.5-coder:0.5b"
VISION_MODEL_NAME = "moondream:1.8b-v2-q4_K_M"

MEMORY_DIR = os.getenv("MEMORY_DIR", "/app/memory")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/app/workspace")
MEMORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json")
CMD_HISTORY_FILE = os.path.join(MEMORY_DIR, "cmd_history.txt")

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

client = Client(host=OLLAMA_HOST)

# Databases
os.makedirs(CHROMA_PATH := os.path.join(MEMORY_DIR, "vector_db"), exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
kb_collection = chroma_client.get_or_create_collection(name="agent_knowledge")
few_shot_collection = chroma_client.get_or_create_collection(name="agent_few_shot")

def get_ollama_embedding(text: str) -> list:
    try:
        return requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text}, timeout=10).json().get('embedding', [])
    except Exception: return []
