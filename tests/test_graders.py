"""
tests/test_graders.py
Tests for server/grading.py — end-of-episode scoring.
Run with: pytest tests/test_graders.py -v
"""

import pytest

from server.grading import grade_episode


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(
    task_id,
    decisions,
    ground_truth,
    incidents=None,
    step_number=None,
    max_steps=10,
    triage_order=None,
):
    """Build a minimal EnvironmentState dict for grader input."""
    agent_links = [d for d in decisions if d.get("action_type") == "link_alerts"]
    triage_ids = [
        d["alert_id"] for d in decisions
        if d.get("action_type") in ("triage", "skip")
    ]
    return {
        "task_id": task_id,
        "seed": 42,
        "step_number": step_number if step_number is not None else max_steps,
        "max_steps": max_steps,
        "done": True,
        "alerts": [],
        "ground_truth": ground_truth,
        "agent_decisions": decisions,
        "agent_links": agent_links,
        "incidents": incidents or [],
        "cumulative_reward": 0.0,
        "grader_score": None,
        "dynamic_alert_ids": set(),
        "triage_order": triage_order if triage_order is not None else triage_ids,
    }


GROUND_TRUTH_5 = [
    {"alert_id": f"alert-{i:03d}", "true_root_cause": "resource_exhaustion",
     "true_severity": "high", "true_remediation": "scale_up", "incident_id": None}
    for i in range(1, 6)
]

PERFECT_DECISIONS_5 = [
    {"alert_id": f"alert-{i:03d}", "action_type": "triage",
     "root_cause": "resource_exhaustion", "severity": "high", "remediation": "scale_up"}
    for i in range(1, 6)
]

ALL_WRONG_DECISIONS_5 = [
    {"alert_id": f"alert-{i:03d}", "action_type": "triage",
     "root_cause": "network_failure", "severity": "low", "remediation": "restart_service"}
    for i in range(1, 6)
]


# ─────────────────────────────────────────────────────────────────────────────
# Easy grader
# ─────────────────────────────────────────────────────────────────────────────

class TestEasyGrader:

    def test_perfect_run_below_1(self):
        """Perfect decisions on easy task → score < 1.0 due to efficiency component."""
        state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5, step_number=5)
        score = grade_episode("easy", state)
        assert 0.85 < score < 1.0

    def test_perfect_run_strictly_between_0_and_1(self):
        """Even a perfect agent produces a grader score strictly in (0, 1)."""
        state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5, step_number=5)
        score = grade_episode("easy", state)
        assert 0.0 < score < 1.0

    def test_all_wrong_scores_near_zero(self):
        """All wrong decisions → score close to 0.0."""
        state = _make_state("easy", ALL_WRONG_DECISIONS_5, GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        assert score < 0.20

    def test_empty_decisions_scores_zero(self):
        """No decisions at all (agent made no moves) → 0.0."""
        state = _make_state("easy", [], GROUND_TRUTH_5, step_number=0)
        score = grade_episode("easy", state)
        assert score == pytest.approx(0.0)

    def test_partial_run_in_range(self):
        """Triaging 3/5 correctly, 2 untriaged → score between 0 and 1."""
        decisions = PERFECT_DECISIONS_5[:3]
        state = _make_state("easy", decisions, GROUND_TRUTH_5, step_number=3)
        score = grade_episode("easy", state)
        assert 0.0 < score < 1.0

    def test_score_always_in_range(self):
        """Grader output is always clamped to [0.0, 1.0]."""
        for decisions in [PERFECT_DECISIONS_5, ALL_WRONG_DECISIONS_5, []]:
            state = _make_state("easy", decisions, GROUND_TRUTH_5)
            score = grade_episode("easy", state)
            assert 0.0 <= score <= 1.0

    def test_severity_partial_credit(self):
        """Severity off by exactly 1 level → 0.50 partial credit per alert."""
        decisions = [
            {"alert_id": f"alert-{i:03d}", "action_type": "triage",
             "root_cause": "resource_exhaustion", "severity": "medium", "remediation": "scale_up"}
            for i in range(1, 6)
        ]
        state = _make_state("easy", decisions, GROUND_TRUTH_5, step_number=5)
        score = grade_episode("easy", state)
        # rc=1.0, sev=0.50, rem=1.0, eff=0.5, ord=1.0 (all same true sev)
        # 0.38*1.0 + 0.28*0.50 + 0.28*1.0 + 0.03*0.5 + 0.03*1.0 = 0.845
        assert score == pytest.approx(0.845, abs=0.02)

    def test_fewer_steps_higher_efficiency(self):
        """Using fewer steps (while still correct) gives a higher score."""
        state_5steps = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5, step_number=5)
        state_8steps = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5, step_number=8)
        assert grade_episode("easy", state_5steps) > grade_episode("easy", state_8steps)

    def test_ordering_affects_score(self):
        """Triaging critical alerts before low-severity ones gives higher ordering score."""
        gt_mixed = [
            {"alert_id": "a1", "true_root_cause": "resource_exhaustion",
             "true_severity": "critical", "true_remediation": "scale_up", "incident_id": None},
            {"alert_id": "a2", "true_root_cause": "resource_exhaustion",
             "true_severity": "low", "true_remediation": "scale_up", "incident_id": None},
            {"alert_id": "a3", "true_root_cause": "resource_exhaustion",
             "true_severity": "critical", "true_remediation": "scale_up", "incident_id": None},
        ]
        good_triage = [
            {"alert_id": a["alert_id"], "action_type": "triage",
             "root_cause": "resource_exhaustion", "severity": a["true_severity"],
             "remediation": "scale_up"}
            for a in gt_mixed
        ]
        good_order = ["a1", "a3", "a2"]
        bad_order = ["a2", "a1", "a3"]

        state_good = _make_state("easy", good_triage, gt_mixed, step_number=3, triage_order=good_order)
        state_bad = _make_state("easy", good_triage, gt_mixed, step_number=3, triage_order=bad_order)

        assert grade_episode("easy", state_good) > grade_episode("easy", state_bad)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestGraderDeterminism:

    def test_same_input_same_output(self):
        state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5)
        assert grade_episode("easy", state) == grade_episode("easy", state)

    def test_different_decisions_different_scores(self):
        perfect_state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5)
        wrong_state = _make_state("easy", ALL_WRONG_DECISIONS_5, GROUND_TRUTH_5)
        assert grade_episode("easy", perfect_state) > grade_episode("easy", wrong_state)


# ─────────────────────────────────────────────────────────────────────────────
# Medium grader
# ─────────────────────────────────────────────────────────────────────────────

_GT_MEDIUM = [
    {"alert_id": "a1", "true_root_cause": "deployment_bug", "true_severity": "high",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a2", "true_root_cause": "deployment_bug", "true_severity": "high",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a3", "true_root_cause": "deployment_bug", "true_severity": "medium",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a4", "true_root_cause": "network_failure", "true_severity": "critical",
     "true_remediation": "escalate_to_team", "incident_id": "inc-2"},
    {"alert_id": "a5", "true_root_cause": "network_failure", "true_severity": "critical",
     "true_remediation": "escalate_to_team", "incident_id": "inc-2"},
    {"alert_id": "a6", "true_root_cause": "false_alarm", "true_severity": "low",
     "true_remediation": "dismiss", "incident_id": None},
]

_PERFECT_TRIAGE_MEDIUM = [
    {"alert_id": "a1", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "high", "remediation": "rollback_deploy"},
    {"alert_id": "a2", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "high", "remediation": "rollback_deploy"},
    {"alert_id": "a3", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "medium", "remediation": "rollback_deploy"},
    {"alert_id": "a4", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "escalate_to_team"},
    {"alert_id": "a5", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "escalate_to_team"},
]

_CORRECT_LINKS_MEDIUM = [
    {"action_type": "link_alerts", "alert_ids": ["a1", "a2", "a3"], "incident_label": "inc-1"},
    {"action_type": "link_alerts", "alert_ids": ["a4", "a5"], "incident_label": "inc-2"},
]

_CORRECT_SKIP_FA = [{"alert_id": "a6", "action_type": "skip"}]


class TestMediumGrader:

    def test_incident_linking_weighted(self):
        """Adding correct links improves the medium score."""
        decisions = _PERFECT_TRIAGE_MEDIUM[:]
        state = _make_state("medium", decisions, _GT_MEDIUM, step_number=5, max_steps=25)
        score_no_links = grade_episode("medium", state)

        decisions_with_links = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM
        state_with_links = _make_state("medium", decisions_with_links, _GT_MEDIUM, step_number=7, max_steps=25)
        score_with_links = grade_episode("medium", state_with_links)

        assert score_with_links > score_no_links

    def test_perfect_triage_no_links_below_1(self):
        """Perfect triage but no incident links → score < 1.0 for medium."""
        decisions = _PERFECT_TRIAGE_MEDIUM[:]
        state = _make_state("medium", decisions, _GT_MEDIUM, max_steps=25)
        score = grade_episode("medium", state)
        assert score < 1.0

    def test_false_alarm_identification(self):
        """Correctly skipping a false alarm improves score."""
        decisions_no_skip = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM
        state_no_skip = _make_state("medium", decisions_no_skip, _GT_MEDIUM, step_number=7, max_steps=25)

        decisions_with_skip = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM + _CORRECT_SKIP_FA
        state_with_skip = _make_state("medium", decisions_with_skip, _GT_MEDIUM, step_number=8, max_steps=25)

        assert grade_episode("medium", state_with_skip) > grade_episode("medium", state_no_skip)

    def test_fa_not_in_classification_denominator(self):
        """False alarm alerts do NOT penalize rc/sev/rem accuracy."""
        # With only triageable alerts handled (not the FA), rc/sev/rem
        # should still be 1.0 over the 5 triageable alerts.
        decisions = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM + _CORRECT_SKIP_FA
        state = _make_state("medium", decisions, _GT_MEDIUM, step_number=8, max_steps=25)
        score = grade_episode("medium", state)
        # All 5 triageable alerts correct + links correct + FA skipped
        # Score should be high (> 0.85)
        assert score > 0.85

    def test_score_in_range(self):
        for decisions in [[], _PERFECT_TRIAGE_MEDIUM, _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM]:
            score = grade_episode("medium", _make_state("medium", decisions, _GT_MEDIUM, max_steps=25))
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Hard grader
# ─────────────────────────────────────────────────────────────────────────────

_GT_HARD = [
    {"alert_id": "h1", "true_root_cause": "config_error", "true_severity": "critical",
     "true_remediation": "fix_config", "incident_id": "stealth-inc"},
    {"alert_id": "h2", "true_root_cause": "config_error", "true_severity": "high",
     "true_remediation": "fix_config", "incident_id": "stealth-inc"},
]

_STEALTH_INCIDENT = [{"incident_id": "stealth-inc", "stealth": True}]

_STEALTH_CORRECT_TRIAGE = [
    {"alert_id": "h1", "action_type": "triage", "root_cause": "config_error",
     "severity": "critical", "remediation": "fix_config"},
    {"alert_id": "h2", "action_type": "triage", "root_cause": "config_error",
     "severity": "high", "remediation": "fix_config"},
]

_STEALTH_WRONG_TRIAGE = [
    {"alert_id": "h1", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "fix_config"},
    {"alert_id": "h2", "action_type": "triage", "root_cause": "network_failure",
     "severity": "high", "remediation": "fix_config"},
]


class TestHardGrader:

    def test_stealth_bonus_applies(self):
        """Hard grader awards a +0.10 bonus for identifying the stealth incident."""
        state_with = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD,
            incidents=_STEALTH_INCIDENT, step_number=2, max_steps=45,
        )
        state_without = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD,
            incidents=[], step_number=2, max_steps=45,
        )
        score_with = grade_episode("hard", state_with)
        score_without = grade_episode("hard", state_without)
        assert score_with > score_without
        assert score_with - score_without == pytest.approx(0.10, abs=0.01)

    def test_no_stealth_incident_no_bonus(self):
        state = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD,
            incidents=[], step_number=2, max_steps=45,
        )
        state_with = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD,
            incidents=_STEALTH_INCIDENT, step_number=2, max_steps=45,
        )
        assert grade_episode("hard", state_with) > grade_episode("hard", state)

    def test_linking_improves_score(self):
        """Adding correct links improves scores for both medium and hard."""
        gt = [
            {"alert_id": "x1", "true_root_cause": "deployment_bug", "true_severity": "high",
             "true_remediation": "rollback_deploy", "incident_id": "i1"},
            {"alert_id": "x2", "true_root_cause": "deployment_bug", "true_severity": "high",
             "true_remediation": "rollback_deploy", "incident_id": "i1"},
        ]
        triage = [
            {"alert_id": "x1", "action_type": "triage", "root_cause": "deployment_bug",
             "severity": "high", "remediation": "rollback_deploy"},
            {"alert_id": "x2", "action_type": "triage", "root_cause": "deployment_bug",
             "severity": "high", "remediation": "rollback_deploy"},
        ]
        link = [{"action_type": "link_alerts", "alert_ids": ["x1", "x2"], "incident_label": "i1"}]

        for task, ms in [("medium", 25), ("hard", 45)]:
            no_link = grade_episode(task, _make_state(task, triage, gt, step_number=2, max_steps=ms))
            with_link = grade_episode(task, _make_state(task, triage + link, gt, step_number=3, max_steps=ms))
            assert with_link > no_link, f"Linking should improve {task} score"

    def test_score_in_range(self):
        for decisions in [[], _STEALTH_CORRECT_TRIAGE, _STEALTH_WRONG_TRIAGE]:
            score = grade_episode("hard", _make_state("hard", decisions, _GT_HARD, max_steps=45))
            assert 0.0 <= score <= 1.0

    def test_grader_never_exactly_1(self):
        decisions = _STEALTH_CORRECT_TRIAGE + [
            {"action_type": "link_alerts", "alert_ids": ["h1", "h2"], "incident_label": "stealth-inc"},
        ]
        state = _make_state(
            "hard", decisions, _GT_HARD,
            incidents=_STEALTH_INCIDENT, step_number=3, max_steps=45,
        )
        score = grade_episode("hard", state)
        assert score < 1.0
        assert score > 0.5

    def test_grader_produces_continuous_values(self):
        scores = set()
        for step_num in [2, 5, 10, 20, 40]:
            state = _make_state(
                "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD,
                incidents=_STEALTH_INCIDENT, step_number=step_num, max_steps=45,
            )
            scores.add(grade_episode("hard", state))
        assert len(scores) >= 3
