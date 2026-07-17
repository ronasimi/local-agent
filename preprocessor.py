import re
import hashlib
import config

def init_few_shot_db():
    if config.few_shot_collection.count() > 0: return
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
        if emb := config.get_ollama_embedding(text):
            doc_id = hashlib.md5(text.encode()).hexdigest()
            config.few_shot_collection.add(ids=[doc_id], embeddings=[emb], documents=[hint])

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
    if not (emb := config.get_ollama_embedding(user_input)) or config.few_shot_collection.count() == 0: 
        return user_input
        
    res = config.few_shot_collection.query(query_embeddings=[emb], n_results=1)
    if res['documents'][0] and (res['distances'][0][0] if res['distances'] else 1.0) < 0.85:
        return f"{user_input}\n\n[CRITICAL HINT: {res['documents'][0][0]}]"
        
    return user_input
