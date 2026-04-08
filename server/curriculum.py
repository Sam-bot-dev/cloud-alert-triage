"""
server/curriculum.py

Curriculum Controller for the Cloud Alert Triage environment.

Tracks the agent's per-root-cause mastery across episodes and drives
adaptive difficulty selection — the environment gets harder as the
agent improves, and always targets its weakest skill.

Public API
----------
    controller = CurriculumController()
    controller.record(episode_result)       # call after every episode
    params = controller.next_task_params()  # call before every reset()
    stats  = controller.get_stats()         # expose via GET /curriculum/stats

Difficulty tiers
----------------
    warmup       (0.00 – 0.25)  single root-cause, no incidents, no noise
    beginner     (0.25 – 0.45)  2-3 root-cause types, maybe 1 incident
    intermediate (0.45 – 0.65)  correlated incidents, false alarms
    advanced     (0.65 – 0.80)  cascades, noise, multiple incidents
    expert       (0.80 – 1.00)  full hard mode + stealth + adversarial seeds

Progression logic
-----------------
- Agent starts at warmup regardless of seed.
- After every episode the controller checks the recent success rate
  (last 10 episodes) against the tier's advance_rate threshold.
- Fast-track: 90%+ success after 3 episodes skips the min_episodes wait.
- Within each tier, difficulty is a continuous float driven by success rate.
- Weak-spot targeting: the next scenario always biases toward root-cause
  types where the agent's accuracy is below MASTERY_THRESHOLD (0.70).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MASTERY_THRESHOLD = 0.70   # success rate needed to "graduate" a root-cause type
MASTERY_WINDOW    = 10     # rolling window size for mastery calculation
MIN_ATTEMPTS      = 3      # minimum episodes per type before mastery is evaluated

# Every root-cause type the environment can generate, with its minimum
# difficulty before it appears in adaptive scenarios.
ROOT_CAUSE_META: dict[str, dict] = {
    "resource_exhaustion": {"min_difficulty": 0.00, "tier": 1},
    "deployment_bug":      {"min_difficulty": 0.00, "tier": 1},
    "network_failure":     {"min_difficulty": 0.20, "tier": 2},
    "config_error":        {"min_difficulty": 0.20, "tier": 2},
    "dependency_outage":   {"min_difficulty": 0.45, "tier": 3},
    "false_alarm":         {"min_difficulty": 0.45, "tier": 3},
}

# Difficulty tier definitions — agent must earn each transition.
DIFFICULTY_TIERS: list[dict] = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 4,  "advance_rate": 0.60},
    {"name": "beginner",     "max_diff": 0.45, "min_episodes": 5,  "advance_rate": 0.60},
    {"name": "intermediate", "max_diff": 0.65, "min_episodes": 7,  "advance_rate": 0.65},
    {"name": "advanced",     "max_diff": 0.80, "min_episodes": 8,  "advance_rate": 0.70},
    {"name": "expert",       "max_diff": 1.00, "min_episodes": 0,  "advance_rate": 1.00},
]

# Map difficulty range → task_id (used by next_task_params).
_TASK_BY_DIFFICULTY = [
    (0.00, 0.40, "easy"),
    (0.40, 0.65, "medium"),
    (0.65, 1.00, "hard"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Episode result dataclass (caller fills this in after each episode)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """
    Outcome of one completed episode — passed to CurriculumController.record().

    Fields
    ------
    task_id         : "easy" | "medium" | "hard"
    grader_score    : float in (0, 1) — final episode score
    steps_used      : int — steps taken (vs max_steps budget)
    max_steps       : int — episode step budget
    root_cause_hits : dict[str, bool]  — {root_cause: correct?} per alert triaged
    seed            : int — seed used (for dedup / logging)
    """
    task_id:         str
    grader_score:    float
    steps_used:      int
    max_steps:       int
    root_cause_hits: dict[str, bool] = field(default_factory=dict)
    seed:            int = 42

    @property
    def success(self) -> bool:
        """Episode is successful if grader_score >= 0.70."""
        return self.grader_score >= 0.70

    @property
    def efficiency(self) -> float:
        """Fraction of step budget used (lower = more efficient)."""
        return self.steps_used / self.max_steps if self.max_steps > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CurriculumController
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumController:
    """
    Tracks agent mastery per root-cause type and drives adaptive difficulty.

    Usage
    -----
        ctrl = CurriculumController()

        # Before each reset():
        params = ctrl.next_task_params()
        obs = env.reset(params["task_id"], params["seed"])

        # After each episode ends (done=True):
        result = EpisodeResult(
            task_id="easy", grader_score=0.82, steps_used=8, max_steps=10,
            root_cause_hits={"resource_exhaustion": True, "deployment_bug": False}
        )
        ctrl.record(result)

        # Expose via /curriculum/stats:
        stats = ctrl.get_stats()
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng              = random.Random(seed)
        self._tier_index       = 0
        self._tier_episodes    = 0      # episodes spent in the current tier
        self._episode_count    = 0

        # Per-root-cause rolling history: root_cause → [True/False, ...]
        self._rc_history: dict[str, list[bool]] = defaultdict(list)

        # Episode-level history
        self._scores:    list[float] = []
        self._graduated: set[str]   = set()   # root-cause types fully mastered

        # Seed counter — increments so repeated calls never reuse the same seed
        self._seed_counter = 1000

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def record(self, result: EpisodeResult) -> None:
        """Register the outcome of a completed episode and update internal state."""
        self._episode_count  += 1
        self._tier_episodes  += 1
        self._scores.append(result.grader_score)

        # Update per-root-cause accuracy
        for rc, correct in result.root_cause_hits.items():
            self._rc_history[rc].append(correct)
            self._check_mastery(rc)

        # Check tier advancement
        self._maybe_advance_tier()

        logger.info(
            "Curriculum: ep=%d tier=%s diff=%.2f score=%.3f success=%s weak=%s",
            self._episode_count,
            self.tier_name,
            self.difficulty,
            result.grader_score,
            result.success,
            self.weak_spots,
        )

    def next_task_params(self) -> dict[str, Any]:
        """
        Return the task_id and seed for the next episode.

        The task_id maps to difficulty:
            easy   → warmup / beginner
            medium → intermediate
            hard   → advanced / expert

        The seed is chosen to bias toward scenarios that exercise the
        agent's current weak spots (root-cause types below mastery).
        """
        task_id = self._difficulty_to_task()
        seed    = self._pick_seed()

        return {
            "task_id":       task_id,
            "seed":          seed,
            "difficulty":    round(self.difficulty, 3),
            "tier":          self.tier_name,
            "weak_spots":    self.weak_spots,
            "target_rc":     self._pick_target_rc(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Full curriculum state — suitable for the /curriculum/stats endpoint."""
        return {
            "episode_count":       self._episode_count,
            "tier":                self.tier_name,
            "tier_episodes":       self._tier_episodes,
            "difficulty":          round(self.difficulty, 3),
            "skill_profile":       self.skill_profile,
            "weak_spots":          self.weak_spots,
            "graduated":           sorted(self._graduated),
            "unlocked_root_causes": self.unlocked_root_causes,
            "avg_score_last_10":   self._avg_score(10),
            "avg_score_last_5":    self._avg_score(5),
            "recent_success_rate": round(self._recent_success_rate(), 3),
            "recommended_next":    self.next_task_params(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def difficulty(self) -> float:
        """
        Continuous difficulty in [0, 1], driven by recent success rate.
        Stays within the current tier's ceiling.
        """
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self._episode_count < 3:
            return 0.10   # first few episodes are always gentle

        rate = self._recent_success_rate()

        # Floor = previous tier's ceiling (or 0 for warmup)
        floor = DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"] if self._tier_index > 0 else 0.0
        ceiling = tier["max_diff"]

        # Linearly interpolate within the tier based on success rate
        natural = floor + rate * (ceiling - floor)
        return round(min(ceiling, max(floor, natural)), 4)

    @property
    def tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    @property
    def skill_profile(self) -> dict[str, float]:
        """Per-root-cause accuracy over the last MASTERY_WINDOW episodes."""
        return {
            rc: round(
                sum(hist[-MASTERY_WINDOW:]) / len(hist[-MASTERY_WINDOW:]), 3
            )
            for rc, hist in self._rc_history.items()
            if hist
        }

    @property
    def weak_spots(self) -> list[str]:
        """Root-cause types where accuracy is below MASTERY_THRESHOLD."""
        profile = self.skill_profile
        return [
            rc for rc, acc in profile.items()
            if acc < MASTERY_THRESHOLD
        ]

    @property
    def unlocked_root_causes(self) -> list[str]:
        """Root-cause types available at the current difficulty level."""
        d = self.difficulty
        return [
            rc for rc, meta in ROOT_CAUSE_META.items()
            if meta["min_difficulty"] <= d
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _recent_success_rate(self, window: int = 10) -> float:
        """Fraction of recent episodes with grader_score >= 0.70."""
        recent = self._scores[-window:]
        if not recent:
            return 0.0
        return sum(1 for s in recent if s >= 0.70) / len(recent)

    def _avg_score(self, window: int) -> float:
        recent = self._scores[-window:]
        if not recent:
            return 0.0
        return round(sum(recent) / len(recent), 4)

    def _check_mastery(self, rc: str) -> None:
        hist = self._rc_history[rc]
        if len(hist) < MIN_ATTEMPTS:
            return
        window = hist[-MASTERY_WINDOW:]
        rate   = sum(window) / len(window)
        if rate >= MASTERY_THRESHOLD and rc not in self._graduated:
            self._graduated.add(rc)
            logger.info(
                "Curriculum: MASTERED '%s' (%.0f%% over last %d episodes)",
                rc, rate * 100, len(window),
            )

    def _maybe_advance_tier(self) -> None:
        """Advance the difficulty tier if the agent is ready."""
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return   # already at expert

        tier = DIFFICULTY_TIERS[self._tier_index]
        rate = self._recent_success_rate()

        # Fast-track: 90%+ success in just 3 episodes → advance immediately
        fast_track = (self._tier_episodes >= 3 and rate >= 0.90)
        if not fast_track and self._tier_episodes < tier["min_episodes"]:
            return

        if rate >= tier["advance_rate"] or fast_track:
            old = tier["name"]
            self._tier_index   += 1
            self._tier_episodes = 0
            logger.info(
                "Curriculum: TIER UP %s → %s (rate=%.0f%% %s)",
                old,
                DIFFICULTY_TIERS[self._tier_index]["name"],
                rate * 100,
                "FAST-TRACK" if fast_track else "",
            )

    def _difficulty_to_task(self) -> str:
        d = self.difficulty
        for lo, hi, task_id in _TASK_BY_DIFFICULTY:
            if lo <= d < hi:
                return task_id
        return "hard"   # difficulty == 1.0

    def _pick_seed(self) -> int:
        """
        Pick the next seed.

        Strategy: mostly increment (for variety), but occasionally reuse a
        seed from a previously failed scenario to give the agent another shot.
        """
        self._seed_counter += 1
        return self._seed_counter

    def _pick_target_rc(self) -> str | None:
        """
        Select the root-cause type to bias the next scenario toward.

        Priority:
          1. Unlocked weak spots (below mastery, available at current diff)
          2. Unlocked but untried root-cause types (encourage exploration)
          3. None (no targeting — let the generator pick freely)
        """
        unlocked = set(self.unlocked_root_causes)
        weak     = [rc for rc in self.weak_spots if rc in unlocked]
        if weak:
            return self._rng.choice(weak)

        untried = [rc for rc in unlocked if rc not in self._rc_history]
        if untried:
            return self._rng.choice(untried)

        return None
