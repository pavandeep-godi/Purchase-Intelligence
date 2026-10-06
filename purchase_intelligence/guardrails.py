"""Deterministic screening and output checks that run around every LLM call."""

from __future__ import annotations

import re

MAX_QUESTION_CHARS = 300

ACTION_PATTERN = re.compile(
    r"\b(?:buy|place (?:an? )?order|order from|purchase from|switch to|sign|approve|pay|award|issue (?:a )?po|cancel|negotiate on my behalf)\b",
    re.IGNORECASE)
INJECTION_PATTERN = re.compile(
    r"ignore (?:all |your |the |any )*(?:previous |prior |above )*(?:rules|instructions|guidelines)|api[ _-]?key|system prompt|"
    r"reveal (?:the |your )?(?:key|prompt|secret|instructions)|secret|password|credential|developer mode|jailbreak|\.env|groq_",
    re.IGNORECASE)

# Prose from an LLM may not contain digits, currency, percentages, or spelled-out amounts.
QUALITATIVE_PATTERN = re.compile(
    r"\d|₹|%|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|"
    r"million|billion|lakh|lakhs|crore|crores|rupees|inr)\b", re.IGNORECASE)
CONTRADICTION_PATTERNS = [
    r"\b(?:data|records|quality) (?:is|are) (?:clean|high|excellent|issue.free)\b",
    r"\b(?:full|complete|perfect) data integrity\b",
    r"\bno (?:critical )?(?:data )?(?:issues?|anomal(?:y|ies)|risks?)\b",
    r"\bwithout (?:any )?(?:issues?|anomal(?:y|ies))\b",
    r"\b(?:guaranteed|certain) savings\b",
    r"\b(?:should|must) (?:buy|switch|order|purchase)\b",
]
# Prose may describe which items appear in the findings, but not assign causes or direction.
CAUSAL_PATTERN = re.compile(
    r"\b(?:driven|drove|drives|due to|because|caused?|attribut\w*|result(?:ed|s)? (?:from|in)|primarily|mainly|thanks to|offset|favou?rable|unfavou?rable|led to)\b",
    re.IGNORECASE)


def screen_question(question: str) -> tuple[str, str]:
    """Return (kind, message): ok, refuse, or unsupported. Runs before any LLM call so it costs no tokens."""
    text = (question or "").strip()
    if not text:
        return "unsupported", "Enter a procurement question."
    if len(text) > MAX_QUESTION_CHARS:
        return "unsupported", f"Questions are limited to {MAX_QUESTION_CHARS} characters."
    if INJECTION_PATTERN.search(text):
        return "refuse", "This request asks about configuration, credentials, or the assistant's rules. Those are never disclosed; the analyst only answers procurement questions about the loaded data."
    if ACTION_PATTERN.search(text):
        return "refuse", ("The analyst cannot place orders, switch suppliers, or approve purchases. "
                          "Ask for an indicative comparison instead, for example 'top savings opportunities' or 'best quotes for Acetone', and review it with your category team.")
    return "ok", ""


def is_safe_prose(text: object, allowed_tokens: list[str], has_quality_issues: bool) -> bool:
    """Accept LLM prose only if it carries no numeric claims and does not contradict known issues or overstate savings."""
    if not isinstance(text, str) or not text.strip():
        return False
    scrubbed = text
    for token in sorted(allowed_tokens, key=len, reverse=True):
        scrubbed = re.sub(re.escape(token), " ", scrubbed, flags=re.IGNORECASE)
    if QUALITATIVE_PATTERN.search(scrubbed) or CAUSAL_PATTERN.search(text):
        return False
    patterns = CONTRADICTION_PATTERNS if has_quality_issues else CONTRADICTION_PATTERNS[4:]
    return not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
