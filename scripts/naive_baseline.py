#!/usr/bin/env python3
"""
naive_baseline.py — Pure LLM baseline without pre-computed hints.

This baseline sends raw alert data to the LLM without:
  - Pre-computed severity hints (sev≈)
  - Cascade group suggestions
  - Priority sorting hints
  - False alarm detection

The LLM must infer everything from the raw observation.
This demonstrates the genuine difficulty of the hard task.
"""

import json
import os
import sys
import time
import httpx
from openai import OpenAI

try:
    from dotenv import load_dotenv
    _sub_env = os.path.join(os.path.dirname(__file__), "cloud-alert-triage", ".env")
    if os.path.exists(_sub_env):
        load_dotenv(_sub_env)
    else:
        load_dotenv()
except ImportError:
    pass

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

_explicit_base = os.environ.get("API_BASE_URL", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")
_openai_key = os.environ.get("OPENAI_API_KEY", "")
_hf_token = os.environ.get("HF_TOKEN")

if _explicit_base:
    API_BASE_URL = _explicit_base
elif _groq_key:
    API_BASE_URL = "https://api.groq.com/openai/v1"
elif _openai_key:
    API_BASE_URL = "https://api.openai.com/v1"
else:
    API_BASE_URL = "https://api-inference.huggingface.co/v1"

if "groq" in API_BASE_URL:
    API_KEY = _groq_key or _hf_token or _openai_key
elif "openai.com" in API_BASE_URL:
    API_KEY = _openai_key or _hf_token
else:
    API_KEY = _hf_token or _openai_key or _groq_key

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required when no other API key is set")

NAIVE_SYSTEM_PROMPT = """\
You are an expert SRE triaging cloud infrastructure alerts.

Analyze the alerts below and decide what action to take for each.
Return a JSON array of actions. Each action must be one of:

1. TRIAGE (for real alerts):
   {"action_type": "triage", "alert_id": "<id>", "root_cause": "...", "severity": "...", "remediation": "..."}

2. SKIP (for false alarms):
   {"action_type": "skip", "alert_id": "<id>"}

3. LINK_ALERTS (for correlated incidents):
   {"action_type": "link_alerts", "alert_ids": ["<id1>", "<id2>", ...], "incident_label": "..."}

IMPORTANT:
- Infer severity from the metric name, value, threshold, and message context
- Infer root cause from the service and metric type
- Use remediation mapping: resource_exhaustion→scale_up, network_failure→escalate_to_team,
  deployment_bug→rollback_deploy, config_error→fix_config, dependency_outage→acknowledge_and_monitor
- Look for false alarm patterns: "scheduled", "maintenance", "P0 auto-created", "prior pattern"
- Look for cascade patterns: alerts on dependent services citing the same upstream

Return ONLY a JSON array, no extra text.
"""

NAIVE_USER_PROMPT = """\
=== ALERTS ===
{alerts}

=== SERVICE DEPENDENCY MAP ===
{service_map}

Decide what actions to take. Return a JSON array.
"""

def fmt_alert_raw(alert: dict) -> str:
    return (
        f"Alert {alert.get('alert_id')}: service={alert.get('service')}, "
        f"metric={alert.get('metric')}, value={alert.get('metric_value')}, "
        f"threshold={alert.get('threshold')}, message={alert.get('message')}, "
        f"context={alert.get('context')}"
    )

def fmt_service_map(sm: dict) -> str:
    lines = []
    for svc, deps in sm.items():
        lines.append(f"  {svc} -> {deps}")
    return "\n".join(lines) if lines else "  (empty)"

def parse_actions(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return []

def run_episode(client: OpenAI, task_id: str, seed: int) -> dict:
    resp = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()["observation"]

    all_actions = []
    steps = 0
    max_steps = obs.get("max_steps", 45)

    while steps < max_steps:
        pending = [a for a in obs.get("alerts", []) if not a.get("triaged")]
        if not pending:
            break

        alerts_text = "\n".join(fmt_alert_raw(a) for a in pending)
        sm_text = fmt_service_map(obs.get("service_map", {}))

        user_prompt = NAIVE_USER_PROMPT.format(
            alerts=alerts_text,
            service_map=sm_text
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=4096,
                timeout=60,
            )
            actions = parse_actions(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}")
            actions = []

        if not actions:
            break

        for action in actions:
            if not action.get("alert_id"):
                continue
            try:
                step_resp = httpx.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30
                )
                step_resp.raise_for_status()
                result = step_resp.json()
                obs = result.get("observation", {})
                all_actions.append(action)
                if result.get("done"):
                    break
            except Exception as e:
                print(f"[WARN] Step failed: {e}")

        steps += 1
        if obs.get("pending_count", 1) == 0:
            break

    return {"steps": steps, "actions": all_actions}

def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "hard"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Running naive baseline: task={task_id}, seed={seed}")
    result = run_episode(client, task_id, seed)
    print(f"Done: steps={result['steps']}, actions={len(result['actions'])}")

if __name__ == "__main__":
    main()
