import re
import hashlib
import config

def init_few_shot_db():
    if config.few_shot_collection.count() > 0: return
    print("Initializing Nomic Dynamic Preprocessor...")
    
    # Casual greetings have been removed from the Vector DB to prevent false-positive tool triggers
    examples = [
        ("What's the weather in London?", "[ACTION REQUIRED: Generate a JSON tool call for 'get_weather'. DO NOT converse.]"),
        ("What is the current wind speed?", "[ACTION REQUIRED: Generate a JSON tool call for 'get_weather'. DO NOT converse.]"),
        ("Who is Microsoft's CEO?", "[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg' or 'search_wikipedia'. DO NOT answer from memory.]"),
        ("Age born on 1990-05-05?", "[ACTION REQUIRED: Generate a JSON tool call for 'calculate_age' and 'get_date_info'. DO NOT calculate manually.]"),
        ("Ping google", "[ACTION REQUIRED: Generate a JSON tool call for 'run_sandboxed_command' with 'ping -c 4 google.com'.]"),
        ("Free RAM?", "[ACTION REQUIRED: Generate a JSON tool call for 'read_system_proc' with 'mem'.]"),
        ("Summarize this site", "[ACTION REQUIRED: Generate a JSON tool call for 'scrape_and_summarize_url'.]"),
        ("Search for recent news on X and summarize it", "[ACTION REQUIRED: Generate a JSON tool call for 'web_research'.]"),
        ("What's out there about Y? Give me sources.", "[ACTION REQUIRED: Generate a JSON tool call for 'web_research'.]"),
        ("What is the largest shark?", "[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg' or 'search_wikipedia'. DO NOT answer from memory.]"),
        ("How far away is the moon?", "[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg' or 'search_wikipedia'. DO NOT answer from memory.]"),
        ("Double check your answer", "[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg'. DO NOT apologize.]"),
        ("Are you sure?", "[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg'. DO NOT apologize.]")
    ]
    for text, hint in examples:
        if emb := config.get_ollama_embedding(text):
            doc_id = hashlib.md5(text.encode()).hexdigest()
            config.few_shot_collection.add(ids=[doc_id], embeddings=[emb], documents=[hint])

def preprocess_user_prompt(user_input: str) -> str:
    # Clean the input to standard lowercase words
    lower_input = re.sub(r'[^a-z\s]', '', user_input.lower().strip())
    words = lower_input.split()
    
    # 1. Comprehensive Greeting & Chit-Chat Escape Hatch
    greeting_keywords = {"hello", "hi", "hey", "yo", "greetings", "sup", "morning", "afternoon", "evening"}
    if words and (words[0] in greeting_keywords or "whats up" in lower_input or "what sup" in lower_input or "how are you" in lower_input or "shakin" in lower_input):
        return user_input
        
    # 2. Hardcoded Keyword Fallbacks 
    verification_triggers = [
        "are you sure", "double check", "verify that", "incorrect", 
        "wrong", "that is false", "not true", "bullshit", "untrue"
    ]
    if any(trigger in lower_input for trigger in verification_triggers):
        return f"{user_input}\n\n[ACTION REQUIRED: Your previous answer is being challenged. DO NOT APOLOGIZE. Generate a JSON tool call for 'search_ddg' immediately to verify.]"
        
    # Combines "wind speed" to ensure isolated commands like "speed up" don't trigger the weather interceptor
    weather_triggers = ["weather", "temperature", "forecast", "wind speed", "conditions"]
    if any(trigger in lower_input for trigger in weather_triggers):
        return f"{user_input}\n\n[ACTION REQUIRED: Generate a JSON tool call for 'get_weather' immediately. Output ONLY the tool call.]"

    # URL and Web Scraping Interceptor
    url_triggers = ["http://", "https://", "www.", "scrape", "fetch", "summarize site"]
    if any(trigger in lower_input for trigger in url_triggers):
        return f"{user_input}\n\n[ACTION REQUIRED: Generate a JSON tool call for 'fetch_url_content' or 'scrape_and_summarize_url'.]"

    # Time and Date Check
    time_triggers = ["what time", "current time", "date is it", "todays date", "what day"]
    if any(trigger in lower_input for trigger in time_triggers):
        return f"{user_input}\n\n[ACTION REQUIRED: Generate a JSON tool call for 'get_system_time' immediately.]"

    # 3. Universal Interrogative (5 Ws + H) Trigger
    question_words = {"who", "what", "where", "when", "why", "how"}
    conversational_exceptions = {"up", "are", "you", "going", "doing", "is", "your", "name", "old", "much"}
    
    if words and words[0] in question_words:
        remaining_words = set(words[1:])
        if not remaining_words.issubset(conversational_exceptions):
            return f"{user_input}\n\n[ACTION REQUIRED: Generate a JSON tool call for 'search_ddg' or 'search_wikipedia' immediately. DO NOT answer from memory.]"

    # 4. Looser Semantic Threshold Fallback
    if not (emb := config.get_ollama_embedding(user_input)) or config.few_shot_collection.count() == 0: 
        return user_input
        
    res = config.few_shot_collection.query(query_embeddings=[emb], n_results=1)
    if res['documents'][0] and (res['distances'][0][0] if res['distances'] else 1.0) < 0.85:
        return f"{user_input}\n\n[ACTION REQUIRED: {res['documents'][0][0]}]"
        
    return user_input
