"""Map local classifier results to constitutional principle scores."""

from __future__ import annotations

import re

from constitution_guard.constitution.principles import LOCAL_PRINCIPLES
from constitution_guard.models import CheckResult, GuardChecks, PrincipleScore

_SCOPE_QUERY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("medical", re.compile(r"\b(diagnos|symptom|medication|prescri|treatment|disease|doctor)\b", re.I)),
    ("legal", re.compile(r"\b(lawsuit|attorney|contract|legal advice|sue|liability)\b", re.I)),
    ("financial", re.compile(r"\b(invest|stock|tax|mortgage|financial advice|portfolio)\b", re.I)),
]

_REFERRAL_PATTERNS = re.compile(
    r"\b(consult|seek|contact|speak with|see a|refer|professional|doctor|attorney|advisor|988)\b",
    re.I,
)


def _check_by_name(results: list[CheckResult], name: str) -> CheckResult | None:
    for r in results:
        if r.name == name:
            return r
    return None


def _score_non_maleficence(results: list[CheckResult]) -> PrincipleScore:
    toxicity = _check_by_name(results, "toxicity")
    crisis = _check_by_name(results, "crisis")
    scores: list[float] = []
    reasons: list[str] = []

    if toxicity:
        scores.append(toxicity.score)
        reasons.append(toxicity.reasoning)
    if crisis:
        crisis_safe = 0.2 if crisis.flagged else 1.0
        scores.append(crisis_safe)
        reasons.append(crisis.reasoning)

    if not scores:
        return PrincipleScore(score=1.0, reasoning="No harm classifiers ran.")

    combined = min(scores)
    return PrincipleScore(
        score=round(combined, 4),
        reasoning="; ".join(reasons),
    )


def _score_from_classifier(results: list[CheckResult], classifier_name: str, label: str) -> PrincipleScore:
    check = _check_by_name(results, classifier_name)
    if check is None:
        return PrincipleScore(score=1.0, reasoning=f"No {label} classifier ran.")
    return PrincipleScore(score=round(check.score, 4), reasoning=check.reasoning)


def _score_scope(query: str, draft: str) -> PrincipleScore:
    domains = [label for label, pattern in _SCOPE_QUERY_PATTERNS if pattern.search(query)]
    if not domains:
        return PrincipleScore(score=1.0, reasoning="Query within general scope.")

    if _REFERRAL_PATTERNS.search(draft):
        return PrincipleScore(
            score=1.0,
            reasoning=f"Professional-domain query ({', '.join(domains)}); response includes referral.",
        )

    return PrincipleScore(
        score=0.5,
        reasoning=f"Professional-domain query ({', '.join(domains)}) without referral to qualified professional.",
    )


def score_local_principles(
    output_checks: GuardChecks, query: str, draft: str
) -> dict[str, PrincipleScore]:
    """Derive local principle scores from classifier output checks."""
    results = output_checks.results
    scores: dict[str, PrincipleScore] = {}

    for principle in LOCAL_PRINCIPLES:
        if principle.local_source == "toxicity":
            scores[principle.name] = _score_non_maleficence(results)
        elif principle.local_source == "pii":
            scores[principle.name] = _score_from_classifier(results, "pii", "PII")
        elif principle.local_source == "jailbreak":
            scores[principle.name] = _score_from_classifier(results, "jailbreak", "jailbreak")
        elif principle.local_source == "heuristic":
            scores[principle.name] = _score_scope(query, draft)

    return scores
