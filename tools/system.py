import os
import glob
import psutil
import docker
import config
from .subagent import sub_agent_task

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
            volumes={os.path.abspath(config.WORKSPACE_DIR): {'bind': '/workspace', 'mode': 'rw'}}, working_dir="/workspace"
        )
        res = out.decode().strip() or "Success (No output)."
        
        if len(res) > 500:
            return sub_agent_task(res, "Extract the specific error message or final outcome from this terminal log.")
        return res[:4000]
    except Exception as e: return f"Execution error: {e}"

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
