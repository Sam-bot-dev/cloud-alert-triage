"""
server/adaptive_scenario.py

Adaptive Scenario Generator for the Cloud Alert Triage environment.

Instead of always using a fixed (task_id, seed) pair, this module generates
scenarios that are *biased toward the agent's current weak spots* — the
root-cause types and incident patterns where it is performing poorly.

Public API
----------
    generator = AdaptiveScenarioGenerator()
    scenario  = generator.generate(task_id, seed, target_rc=None, weak_spots=[])

How it works
------------
1. Calls the base generate_scenario(task_id, seed) to get a deterministic
   scenario as a starting point.
2. If `target_rc` is set (a root-cause type the agent is struggling with),
   swaps a random subset of the scenario's alerts to use that root-cause —
   while keeping the scenario structurally valid (timestamps, service graph,
   metric names, incident links all remain consistent).
3. If `weak_spots` is provided but `target_rc` is not, one weak spot is
   chosen at random and used as the target.
4. If neither is set, the scenario is returned unchanged (identical to the
   base generator).

The adaptive layer is purely additive — it never breaks existing
determinism contracts when target_rc is None.

Design constraints
------------------
- All mutations preserve alert_id stability (alert-001 stays alert-001).
- Ground truth is rewritten to match any swapped root-causes.
- Incident links are preserved; only non-incident alerts are swapped.
- Swapped alerts get realistic metric/message/context matching the new
  root-cause so the change is meaningful, not cosmetic.
- The adaptive layer is transparent to the grader: ground_truth is
  authoritative and always matches the generated scenario.
"""

from __future__ import annotations

import random
from typing import Any

from server.scenario_generator import generate_scenario
from server.service_graph import get_graph_as_adjacency_list

# ─────────────────────────────────────────────────────────────────────────────
# Root-cause → realistic (metric, threshold, message_template, remediation)
# ─────────────────────────────────────────────────────────────────────────────

_RC_TEMPLATES: dict[str, list[dict]] = {
    "resource_exhaustion": [
        {
            "metric": "cpu_usage_percent",
            "threshold": 80.0,
            "value_range": (82.0, 98.0),
            "msg": "{service} CPU usage at {value:.1f}% — above {threshold:.0f}% threshold",
            "context": None,
            "remediation": "scale_up",
        },
        {
            "metric": "memory_usage_percent",
            "threshold": 85.0,
            "value_range": (87.0, 99.0),
            "msg": "{service} memory usage at {value:.1f}% — OOM risk",
            "context": None,
            "remediation": "scale_up",
        },
    ],
    "deployment_bug": [
        {
            "metric": "error_rate_percent",
            "threshold": 5.0,
            "value_range": (8.0, 35.0),
            "msg": "{service} error rate {value:.1f}% following recent deploy",
            "context": "Recent deploy 47 minutes ago. Rollback candidate.",
            "remediation": "rollback_deploy",
        },
        {
            "metric": "health_check_failures",
            "threshold": 3.0,
            "value_range": (4.0, 12.0),
            "msg": "{service} health checks failing after deploy",
            "context": "New image deployed 2 hours ago. Service unhealthy.",
            "remediation": "rollback_deploy",
        },
    ],
    "network_failure": [
        {
            "metric": "network_latency_ms",
            "threshold": 200.0,
            "value_range": (250.0, 800.0),
            "msg": "{service} network latency {value:.0f}ms — above {threshold:.0f}ms SLA",
            "context": None,
            "remediation": "escalate_to_team",
        },
        {
            "metric": "packet_loss_percent",
            "threshold": 5.0,
            "value_range": (6.0, 25.0),
            "msg": "{service} packet loss {value:.1f}% — network instability",
            "context": None,
            "remediation": "escalate_to_team",
        },
    ],
    "config_error": [
        {
            "metric": "auth_failure_rate",
            "threshold": 0.05,
            "value_range": (0.08, 0.50),
            "msg": "{service} auth failures spiking — likely misconfigured credentials",
            "context": "Config change 30 minutes ago. Env var mismatch suspected.",
            "remediation": "fix_config",
        },
        {
            "metric": "connection_refused_count",
            "threshold": 10.0,
            "value_range": (12.0, 60.0),
            "msg": "{service} refusing connections — check service configuration",
            "context": "Port mapping changed in last deployment.",
            "remediation": "fix_config",
        },
    ],
    "dependency_outage": [
        {
            "metric": "upstream_error_rate",
            "threshold": 10.0,
            "value_range": (15.0, 60.0),
            "msg": "{service} upstream errors — dependency may be down",
            "context": "Upstream service showing degraded health.",
            "remediation": "acknowledge_and_monitor",
        },
        {
            "metric": "dependency_timeout_count",
            "threshold": 5.0,
            "value_range": (7.0, 40.0),
            "msg": "{service} dependency timeouts — cascading from upstream failure",
            "context": None,
            "remediation": "acknowledge_and_monitor",
        },
    ],
    "false_alarm": [
        {
            "metric": "cpu_usage_percent",
            "threshold": 80.0,
            "value_range": (72.0, 79.5),
            "msg": "{service} CPU usage elevated — monitoring system over-triggered",
            "context": "PagerDuty P0 auto-created — prior pattern suggests false positive.",
            "remediation": "dismiss",
        },
    ],
}

# Severity mapping for each root cause (used when swapping)
_RC_SEVERITY: dict[str, str] = {
    "resource_exhaustion": "high",
    "deployment_bug":      "high",
    "network_failure":     "high",
    "config_error":        "medium",
    "dependency_outage":   "medium",
    "false_alarm":         "low",
}


class AdaptiveScenarioGenerator:
    """
    Wraps the base scenario generator and applies targeted mutations based
    on the agent's current weak spots.

    Usage
    -----
        gen = AdaptiveScenarioGenerator()
        scenario = gen.generate(
            task_id="medium",
            seed=1042,
            target_rc="config_error",      # the RC type to inject more of
            weak_spots=["config_error"],   # used if target_rc is None
        )
    """

    def __init__(self) -> None:
        self._service_map = get_graph_as_adjacency_list()

    def generate(
        self,
        task_id: str,
        seed: int,
        target_rc: str | None = None,
        weak_spots: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a scenario, optionally biased toward `target_rc`.

        Parameters
        ----------
        task_id   : "easy" | "medium" | "hard"
        seed      : int — base seed for determinism
        target_rc : root-cause type to over-represent (optional)
        weak_spots: fallback list — one is chosen if target_rc is None

        Returns
        -------
        Same dict structure as generate_scenario(): alerts, ground_truth, incidents.
        The adaptive_metadata key is added for transparency.
        """
        # Resolve target root-cause
        chosen_rc = target_rc
        if chosen_rc is None and weak_spots:
            rng = random.Random(seed + 9999)
            # Pick the most actionable weak spot that has templates
            available = [rc for rc in weak_spots if rc in _RC_TEMPLATES]
            if available:
                chosen_rc = rng.choice(available)

        # Generate the base scenario
        scenario = generate_scenario(task_id, seed)

        if chosen_rc is None or chosen_rc not in _RC_TEMPLATES:
            # No targeting — return base scenario unchanged
            scenario["adaptive_metadata"] = {
                "targeted": False,
                "target_rc": None,
                "alerts_swapped": 0,
            }
            return scenario

        # Apply adaptive mutation
        scenario, swapped = self._inject_target_rc(scenario, chosen_rc, seed)
        scenario["adaptive_metadata"] = {
            "targeted": True,
            "target_rc": chosen_rc,
            "alerts_swapped": swapped,
        }
        return scenario

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive mutation logic
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_target_rc(
        self,
        scenario: dict[str, Any],
        target_rc: str,
        seed: int,
    ) -> tuple[dict[str, Any], int]:
        """
        Swap a subset of non-incident alerts to use `target_rc`.

        Rules:
        - Only alerts that are NOT part of a true incident are candidates.
        - Never swap false-alarm alerts (they serve a structural role).
        - Swap between 1 and 30% of eligible alerts (at least 1 if eligible).
        - For each swapped alert, rewrite: metric, metric_value, threshold,
          message, context, and ground_truth row.
        - Severity is updated to match the new root cause.
        """
        rng = random.Random(seed + 42)

        alerts      = scenario["alerts"]
        ground_truth = scenario["ground_truth"]
        incidents   = scenario["incidents"]

        # Build set of alert_ids that are part of true incidents (protected)
        incident_alert_ids: set[str] = set()
        for inc in incidents:
            for aid in inc.get("alert_ids", []):
                incident_alert_ids.add(aid)

        # Build ground-truth lookup
        gt_by_id: dict[str, dict] = {gt["alert_id"]: gt for gt in ground_truth}

        # Find eligible alerts: not in incident, not already the target RC
        eligible = [
            a for a in alerts
            if a["alert_id"] not in incident_alert_ids
            and gt_by_id.get(a["alert_id"], {}).get("true_root_cause") != target_rc
            and gt_by_id.get(a["alert_id"], {}).get("true_root_cause") != "false_alarm"
        ]

        if not eligible:
            return scenario, 0

        # Swap between 1 and 30% of eligible (at least 1)
        n_swap = max(1, int(len(eligible) * 0.30))
        to_swap = rng.sample(eligible, min(n_swap, len(eligible)))

        templates = _RC_TEMPLATES[target_rc]
        swapped   = 0

        for alert in to_swap:
            aid      = alert["alert_id"]
            service  = alert["service"]
            tmpl     = rng.choice(templates)
            lo, hi   = tmpl["value_range"]
            value    = round(rng.uniform(lo, hi), 2)

            # Rewrite alert fields
            alert["metric"]        = tmpl["metric"]
            alert["metric_value"]  = value
            alert["threshold"]     = tmpl["threshold"]
            alert["message"]       = tmpl["msg"].format(
                service=service, value=value, threshold=tmpl["threshold"]
            )
            alert["context"] = tmpl["context"]

            # Rewrite ground truth
            if aid in gt_by_id:
                severity = _RC_SEVERITY.get(target_rc, "medium")
                gt_by_id[aid]["true_root_cause"]  = target_rc
                gt_by_id[aid]["true_severity"]     = severity
                gt_by_id[aid]["true_remediation"]  = tmpl["remediation"]

            swapped += 1

        # Reconstruct ground_truth list preserving order
        scenario["ground_truth"] = [gt_by_id[a["alert_id"]] for a in alerts if a["alert_id"] in gt_by_id]

        return scenario, swapped
