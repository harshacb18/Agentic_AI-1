"""Transparent intent routing and retrieval policy for the interview coach."""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    INTERVIEW = "interview"
    EVALUATE = "evaluate"
    COACH = "coach"
    REPORT = "report"


_INTENT_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (Intent.REPORT, re.compile(r"\b(report|summary|summarize|session recap)\b", re.I)),
    (
        Intent.EVALUATE,
        re.compile(r"\b(evaluate|score|assess|rate|review my answer|grade)\b", re.I),
    ),
    (
        Intent.COACH,
        re.compile(r"\b(coach|improve|feedback|weakness|how can i|help me)\b", re.I),
    ),
    (
        Intent.INTERVIEW,
        re.compile(r"\b(interview|practice|ask me|next question|start)\b", re.I),
    ),
)


def detect_intent(message: str, *, has_active_question: bool) -> Intent:
    """Route an explicit request; treat an unlabelled interview turn as an answer."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message):
            return intent
    return Intent.EVALUATE if has_active_question else Intent.INTERVIEW


def should_retrieve(intent: Intent, *, company: str, role: str, has_evaluations: bool) -> bool:
    """Only call the vector store when company evidence can help the requested task."""
    if not company.strip() or not role.strip():
        return False
    if intent is Intent.REPORT:
        return has_evaluations
    return intent in {Intent.INTERVIEW, Intent.EVALUATE, Intent.COACH}
