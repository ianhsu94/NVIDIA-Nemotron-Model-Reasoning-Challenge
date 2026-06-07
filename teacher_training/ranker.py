from __future__ import annotations

from typing import Any


SOLVER_PRIOR_DEFAULT = 0.5


def rank_cipher(results: list[dict[str, Any]], solver_prior: dict[str, float] | None = None) -> dict[str, Any]:
    return _rank(results, solver_prior or {}, example_weight=0.60, confidence_weight=0.20, simplicity_weight=0.10, prior_weight=0.10)


def rank_symbol(results: list[dict[str, Any]], solver_prior: dict[str, float] | None = None) -> dict[str, Any]:
    return _rank(results, solver_prior or {}, example_weight=0.70, confidence_weight=0.10, simplicity_weight=0.15, prior_weight=0.05)


def _rank(
    results: list[dict[str, Any]],
    solver_prior: dict[str, float],
    example_weight: float,
    confidence_weight: float,
    simplicity_weight: float,
    prior_weight: float,
) -> dict[str, Any]:
    if not results:
        return {
            "answer": "",
            "confidence": 0.0,
            "matched_examples": 0,
            "total_examples": 0,
            "solver_name": "NoSolver",
            "reason": "no result",
            "score": 0.0,
        }
    best = None
    for result in results:
        total = max(int(result.get("total_examples", 0)), 1)
        match_rate = int(result.get("matched_examples", 0)) / total
        confidence = float(result.get("confidence", 0.0))
        simplicity = float(result.get("simplicity_score", 0.5))
        prior = float(solver_prior.get(result.get("solver_name", ""), SOLVER_PRIOR_DEFAULT))
        score = example_weight * match_rate + confidence_weight * confidence + simplicity_weight * simplicity + prior_weight * prior
        result = dict(result)
        result["score"] = score
        if best is None or score > best["score"]:
            best = result
    return best or results[0]

