from .web import search_ddg, search_wikipedia, search_imdb, web_research, fetch_url_content, scrape_and_summarize_url, ingest_url_to_knowledge_base, crawl_and_ingest_domain
from .system import read_system_proc, get_docker_info, run_sandboxed_command, manage_processes
from .media import read_pdf, analyze_image
from .database import query_mariadb
from .helpers import get_weather, read_file, write_file, list_workspace_files, get_system_time, get_date_info, calculate_age, query_knowledge_base

# Dynamic bindings
AVAILABLE_TOOLS = [
    search_ddg, search_wikipedia, search_imdb, web_research, get_weather, read_file, write_file, read_pdf, analyze_image,
    read_system_proc, get_docker_info, run_sandboxed_command, list_workspace_files,
    fetch_url_content, scrape_and_summarize_url, manage_processes, query_mariadb,
    get_system_time, get_date_info, calculate_age, query_knowledge_base,
    ingest_url_to_knowledge_base, crawl_and_ingest_domain
]
AVAILABLE_FUNCTIONS = {f.__name__: f for f in AVAILABLE_TOOLS}
