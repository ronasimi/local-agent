import json
import pymysql
import config
from .subagent import sub_agent_task

def query_mariadb(sql_query: str) -> str:
    """Execute read-only SELECT queries on the database."""
    if not all([config.DB_HOST, config.DB_USER, config.DB_PASS, config.DB_NAME]):
        return "Database not configured. Set DB_HOST, DB_USER, DB_PASS, and DB_NAME environment variables to enable this tool."
    if not sql_query.strip().upper().startswith("SELECT"): return "Error: Only SELECT permitted."
    try:
        with pymysql.connect(host=config.DB_HOST, user=config.DB_USER, password=config.DB_PASS, database=config.DB_NAME, cursorclass=pymysql.cursors.DictCursor) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchmany(10)
                raw_json = json.dumps(rows, indent=2) if rows else "0 rows returned."
                
                if rows:
                    return sub_agent_task(raw_json, "Translate this raw JSON SQL output into a clean, concise bulleted text summary.")
                return raw_json
    except Exception as e: return f"DB error: {e}"
