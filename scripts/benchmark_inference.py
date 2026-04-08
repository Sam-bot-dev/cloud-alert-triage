#!/usr/bin/env python3
"""
Benchmark inference.py across tasks/seeds, reporting real grader scores.
Usage: python scripts/benchmark_inference.py [task] [--seed SEED] [--api-key KEY]
"""

import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List

import httpx

SERVER_CMD = ["uvicorn", "server.app:app", "--port", "7860", "--host", "0.0.0.0"]
DEFAULT_TASKS = ["easy", "medium", "hard"]
DEFAULT_SEEDS = [42, 123, 456]
API_KEY_VAR = "OPENAI_API_KEY"

def start_server() -> subprocess.Popen:
    """Start server in background, wait ready."""
    proc = subprocess.Popen(
        SERVER_CMD,
        cwd="cloud-alert-triage",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(3)  # Wait startup
    # Health check
    try:
        httpx.get("http://localhost:7860/health", timeout=5)
    except:
        raise RuntimeError("Server failed to start")
    return proc

def run_inference(task: str, seed: int, api_key: str) -> Dict[str, float]:
    """Run inference.py for single task, parse [END] line for score."""
    env = os.environ.copy()
    env["TASK_ID"] = task
    env["SEED"] = str(seed)
    if api_key:
        env[API_KEY_VAR] = api_key
    env["ENV_URL"] = "http://localhost:7860"

    # Modify inference.py temporarily for single-task if needed
    orig_inference = "inference.py"
    single_inference = f"{orig_inference}.single"
    
    proc = subprocess.run(
        ["python", orig_inference],
        cwd="cloud-alert-triage",
        env=env,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    # Parse logs for [END] score
    score_match = re.search(r'\[END\].*score=([\d.]+)', proc.stdout)
    steps_match = re.search(r'\[END\].*steps=(\d+)', proc.stdout)
    
    if score_match:
        return {"score": float(score_match.group(1)), "steps": int(steps_match.group(1)) if steps_match else 0}
    raise ValueError(f"No score in output: {proc.stdout[-500:]}")

def main():
    tasks = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_TASKS
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    server = start_server()
    try:
        results = defaultdict(list)
        
        for task in tasks:
            for seed in DEFAULT_SEEDS:
                print(f"Running {task} seed={seed}...")
                result = run_inference(task, seed, api_key)
                results[task].append(result["score"])
                print(f"  Score: {result['score']:.3f}")
        
        print("\n=== BENCHMARK RESULTS ===")
        for task, scores in results.items():
            mean = sum(scores) / len(scores)
            std = (sum((s - mean)**2 for s in scores) / len(scores))**0.5
            print(f"{task.capitalize()}: {mean:.3f} ± {std:.3f} (n={len(scores)})")
        
        overall_mean = sum(sum(results[t]) for t in results) / sum(len(results[t]) for t in results)
        print(f"Overall: {overall_mean:.3f}")
    
    finally:
        server.terminate()

if __name__ == "__main__":
    main()
