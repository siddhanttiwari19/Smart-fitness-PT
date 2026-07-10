from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import List


class SessionLogger:
    """Tracks reps, scores, and corrections during a squat session."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self.corrections: dict[str, int] = defaultdict(int)
        self.rep_scores: list[float]     = []
        self.rep_details: list[dict]     = []   # per-rep drill-down

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return time.monotonic() - self._start

    @property
    def total_reps(self) -> int:
        return len(self.rep_scores)

    @property
    def total_corrections(self) -> int:
        return sum(self.corrections.values())

    @property
    def avg_score(self) -> float:
        return sum(self.rep_scores) / len(self.rep_scores) if self.rep_scores else 0.0

    @property
    def most_frequent_issue(self) -> str | None:
        if not self.corrections:
            return None
        return max(self.corrections.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------
    def record_correction(self, issue_key: str) -> None:
        self.corrections[issue_key] += 1

    def record_rep(self, rep_number: int, score: float, issues: List[str], depth_ok: bool) -> None:
        self.rep_scores.append(score)
        self.rep_details.append({
            "rep": rep_number,
            "score": round(score, 1),
            "issues": issues,
            "depth_ok": depth_ok,
        })

    # ------------------------------------------------------------------
    def save(self, output_file: str = "session_log.json") -> None:
        """Append a session entry to *output_file* as a JSON array."""
        entry = {
            "date":              datetime.now().isoformat(timespec="seconds"),
            "duration_seconds":  round(self.duration, 1),
            "total_reps":        self.total_reps,
            "avg_score":         round(self.avg_score, 1),
            "total_corrections": self.total_corrections,
            "corrections_by_type": dict(self.corrections),
            "most_frequent_issue": self.most_frequent_issue,
            "rep_details":       self.rep_details,
        }

        log: list = []
        if os.path.exists(output_file):
            try:
                with open(output_file) as fh:
                    log = json.load(fh)
            except (json.JSONDecodeError, ValueError):
                log = []

        log.append(entry)
        with open(output_file, "w") as fh:
            json.dump(log, fh, indent=2)

        print(f"\nSession saved → {output_file}")
        print(
            f"  Duration    : {self.duration / 60:.1f} min  |  "
            f"Reps: {self.total_reps}  |  "
            f"Avg score: {self.avg_score:.1f}  |  "
            f"Corrections: {self.total_corrections}"
        )
        if self.most_frequent_issue:
            print(f"  Top issue   : {self.most_frequent_issue}")
