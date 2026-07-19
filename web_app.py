import os
import json
import streamlit as st

# Central imports from your agent architecture
import config
from tools import AVAILABLE_FUNCTIONS
from preprocessor import init_few_shot_db, preprocess_user_prompt
from agent import execute_react_loop, REACT_SYSTEM_PROMPT, save_memory

st.set_page_config(page_title="Local Agent UI", layout="wide")

# Inject Custom CSS exclusively for Chat Icons
st.markdown("""
<style>
    /* User avatar (Blue - *.color4) */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #8ab4f8 !important;
    }
    /* Assistant avatar (Orange/Yellow - *.color3) */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: #f4bf75 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title(f"Local Agent ({config.MODEL_NAME})")

# 1. Initialize environment (Runs once per server boot)
@st.cache_resource
def setup_environment():
    # Ensure all required models exist locally
    for model in [config.MODEL_NAME, config.SUB_MODEL_NAME, config.VISION_MODEL_NAME]:
        try:
            config.client.show(model)
        except Exception:
            print(f"Pulling missing model: {model}...")
            config.client.pull(model)
            
    # Initialize Vector DBs
    init_few_shot_db()
    return True

setup_environment()

# 2. Session State Initialization
if "messages" not in st.session_state:
    if os.path.exists(config.MEMORY_FILE):
        with open(config.MEMORY_FILE, 'r') as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]

if "verbose_mode" not in st.session_state:
    st.session_state.verbose_mode = False

# New States for File Tracking and Proactive UI Injection
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "auto_prompt" not in st.session_state:
    st.session_state.auto_prompt = None

# 3. Sidebar Setup
with st.sidebar:
    st.header("Agent Controls")
    
    # Think option (Native Toggle Binding)
    st.toggle(
        "Verbose Mode (/think)", 
        key="verbose_mode",
        help="Enables dense terminal logging for agent chain-of-thought analysis."
    )
    st.caption(f"Status: {'ON' if st.session_state.verbose_mode else 'OFF'}")
    
    st.markdown("---")
    
    # Workspace File Uploader
    st.subheader("Workspace Upload")
    uploaded_files = st.file_uploader(
        "Upload files or images to the agent's workspace", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Filter for strictly new files to prevent infinite notification loops
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            os.makedirs(config.WORKSPACE_DIR, exist_ok=True)
            filenames = []
            for uploaded_file in new_files:
                file_path = os.path.join(config.WORKSPACE_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                filenames.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                
            st.success(f"Saved {len(new_files)} file(s) to workspace!")
            
            # --- Proactive Agent Trigger ---
            # Set a ghost prompt that the agent will read immediately
            file_names_str = ", ".join(filenames)
            st.session_state.auto_prompt = f"I just uploaded the following file(s) to the workspace: {file_names_str}. Please review them and tell me how you can assist me with them."
            
    st.markdown("---")
    
    # Wipe Chat History Button
    if st.button("Wipe Memory (/wipe)", use_container_width=True, type="secondary"):
        st.session_state.messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        st.session_state.processed_files.clear() # Clear file tracker on memory wipe
        save_memory(st.session_state.messages)
        st.toast("🧹 Chat memory wiped successfully!")
        st.rerun()
        
    # Wipe Knowledge Base Button
    if st.button("Wipe Vector DB (/wipe_kb)", use_container_width=True, type="primary"):
        with st.spinner("Resetting databases..."):
            for coll_name in ["agent_knowledge", "agent_few_shot"]:
                try: 
                    config.chroma_client.delete_collection(coll_name)
                except Exception: 
                    pass
            try:
                config.kb_collection = config.chroma_client.get_or_create_collection("agent_knowledge")
                config.few_shot_collection = config.chroma_client.get_or_create_collection("agent_few_shot")
                init_few_shot_db()
                st.toast("🧠 KB & Few-Shot databases completely reset!")
                st.rerun()
            except Exception as e: 
                st.error(f"Error resetting database: {e}")
                
    st.markdown("---")
    
    # Available Tools Inventory List
    st.subheader(f"Active Tools ({len(AVAILABLE_FUNCTIONS)})")
    for tool_name in sorted(AVAILABLE_FUNCTIONS.keys()):
        st.markdown(f"- `{tool_name}`")

# 4. Render Main Chat History Window
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. User Input Action and ReAct Execution
user_input = st.chat_input("Message your agent...")

# Intercept the input loop if an automated upload prompt is waiting
prompt = user_input or st.session_state.auto_prompt

if prompt:
    # Immediately clear the flag to prevent looping
    if st.session_state.auto_prompt:
        st.session_state.auto_prompt = None
        
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Apply nomic preprocessor logic to incoming prompts
    enriched_input = preprocess_user_prompt(prompt)
    st.session_state.messages.append({"role": "user", "content": enriched_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking and parsing tools..."):
            response_text, st.session_state.messages = execute_react_loop(
                st.session_state.messages, 
                verbose=st.session_state.verbose_mode
            )
            
            # Convert terminal color codes to Markdown backticks
            clean_text = response_text.replace('\033[90m', '`').replace('\033[0m', '`')
            
            st.markdown(clean_text)
            
    save_memory(st.session_state.messages)
    
    # If this was triggered by the UI (not typing), force a clean refresh
    if not user_input:
        st.rerun()
