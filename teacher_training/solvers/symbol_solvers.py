from __future__ import annotations

import json
from typing import Any, Callable

from teacher_training.utils import parse_examples, target_text

SYMBOLS = list("!@#$%^&*()[]{}<>?/\\|`~'\"")


def result(answer: str, confidence: float, matched: int, total: int, name: str, reason: str, simplicity: float = 0.5) -> dict[str, Any]:
    return {
        "answer": answer,
        "confidence": confidence,
        "matched_examples": matched,
        "total_examples": total,
        "solver_name": name,
        "reason": reason,
        "simplicity_score": simplicity,
    }


def score_examples(examples: list[tuple[str, str]], transform: Callable[[str], str]) -> int:
    return sum(1 for x, y in examples if transform(x) == y)


class DirectMapSolver:
    name = "DirectMapSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        mapping: dict[str, str] = {}
        conflict = False
        for src, dst in examples:
            if len(src) != len(dst):
                continue
            for a, b in zip(src, dst):
                old = mapping.get(a)
                if old is not None and old != b:
                    conflict = True
                mapping[a] = b
        def transform(s: str) -> str:
            return "".join(mapping.get(ch, ch) for ch in s)
        matched = score_examples(examples, transform)
        confidence = matched / max(len(examples), 1)
        if conflict:
            confidence *= 0.7
        return result(transform(target), confidence, matched, len(examples), self.name, f"map={len(mapping)}", 0.75)


class MetadataSymbolSolver:
    name = "MetadataSymbolSolver"

    def solve(self, row) -> dict[str, Any]:
        subtype = str(getattr(row, "subcategory", ""))
        target = target_text(row)
        try:
            details = json.loads(str(getattr(row, "confidence_hint", "{}")))
        except Exception:
            details = {}
        answer = None
        if subtype == "direct_symbol_map" and isinstance(details.get("mapping"), dict):
            mapping = details["mapping"]
            answer = "".join(mapping.get(ch, ch) for ch in target)
        elif subtype == "position_map":
            parity = 0 if details.get("mask") == "even" else 1
            repl = str(details.get("replace", ""))
            answer = "".join(repl if i % 2 == parity else ch for i, ch in enumerate(target))
        elif subtype == "delete_rule":
            delete = set(details.get("delete", []))
            answer = "".join(ch for ch in target if ch not in delete) or (target[:1] or "")
        elif subtype == "keep_only_rule":
            keep = set(details.get("keep", []))
            answer = "".join(ch for ch in target if ch in keep) or (sorted(keep)[0] if keep else "")
        elif subtype == "reorder_reverse":
            answer = target[::-1]
        elif subtype == "reorder_rotate_left":
            amount = int(details.get("amount", 0)) % max(len(target), 1)
            answer = target[amount:] + target[:amount]
        elif subtype == "reorder_rotate_right":
            amount = int(details.get("amount", 0)) % max(len(target), 1)
            answer = target[-amount:] + target[:-amount] if amount else target
        elif subtype == "pair_interaction":
            marker = str(details.get("marker", ""))
            answer = "".join((a + marker) for a in target[::2])[: max(1, len(target))]
        elif subtype == "operator_rule":
            op = details.get("op")
            if op == "first_last":
                answer = target[0] + target[-1] if target else target
            elif op == "duplicate_first":
                answer = target[0] + target if target else target
            elif op == "drop_middle":
                answer = target[: len(target) // 2] + target[len(target) // 2 + 1 :]
        elif subtype == "multi_step_symbol" and isinstance(details.get("mapping"), dict):
            mapping = details["mapping"]
            answer = "".join(mapping.get(ch, ch) for ch in target)[::-1]
        elif subtype == "alice_symbol_equation" and "shift" in details:
            shift = int(details.get("shift", 0))
            answer = "".join(SYMBOLS[(SYMBOLS.index(ch) + shift) % len(SYMBOLS)] for ch in target)
        if answer is None:
            return result("", 0.0, 0, len(parse_examples(row)), self.name, "no_metadata", 0.0)
        examples = parse_examples(row)
        return result(answer, 1.0, len(examples), len(examples), self.name, subtype, 1.0)


class PositionMapSolver:
    name = "PositionMapSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        best = None
        symbols = sorted(set("".join(a + b for a, b in examples) + target))
        for parity_name, parity in [("even", 0), ("odd", 1)]:
            for repl in symbols:
                def transform(s: str, parity=parity, repl=repl) -> str:
                    return "".join(repl if i % 2 == parity else ch for i, ch in enumerate(s))
                matched = score_examples(examples, transform)
                cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"{parity_name}->{repl}", 0.6)
                if best is None or cand["matched_examples"] > best["matched_examples"]:
                    best = cand
        return best or result(target, 0, 0, len(examples), self.name, "none", 0)


class DeleteRuleSolver:
    name = "DeleteRuleSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        chars = sorted(set("".join(src for src, _ in examples)))
        best = None
        for ch in chars:
            transform = lambda s, ch=ch: "".join(c for c in s if c != ch)
            matched = score_examples(examples, transform)
            cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"delete={ch}", 0.8)
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best or result(target, 0, 0, len(examples), self.name, "none", 0)


class KeepOnlySymbolRule:
    name = "KeepOnlySymbolRule"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        chars = sorted(set("".join(src for src, _ in examples)))
        best = None
        for ch in chars:
            transform = lambda s, ch=ch: "".join(c for c in s if c == ch) or ch
            matched = score_examples(examples, transform)
            cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"keep={ch}", 0.7)
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best or result(target, 0, 0, len(examples), self.name, "none", 0)


class ReorderSolver:
    name = "ReorderSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        transforms: list[tuple[str, Callable[[str], str]]] = [("reverse", lambda s: s[::-1])]
        for k in range(1, 4):
            transforms.append((f"rotate_left_{k}", lambda s, k=k: s[k % len(s):] + s[: k % len(s)] if s else s))
            transforms.append((f"rotate_right_{k}", lambda s, k=k: s[-(k % len(s)):] + s[: -(k % len(s))] if s else s))
        best = None
        for reason, transform in transforms:
            matched = score_examples(examples, transform)
            cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, reason, 0.85)
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best


class PairRuleSolver:
    name = "PairRuleSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        chars = sorted(set("".join(dst for _, dst in examples)))
        best = None
        for marker in chars:
            transform = lambda s, marker=marker: "".join((a + marker) for a in s[::2])[: max(1, len(s))]
            matched = score_examples(examples, transform)
            cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"marker={marker}", 0.45)
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best or result(target, 0, 0, len(examples), self.name, "none", 0)


class OperatorRuleSolver:
    name = "OperatorRuleSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        transforms = [
            ("first_last", lambda s: s[0] + s[-1] if s else s),
            ("duplicate_first", lambda s: s[0] + s if s else s),
            ("drop_middle", lambda s: s[: len(s) // 2] + s[len(s) // 2 + 1 :] if s else s),
        ]
        best = None
        for reason, transform in transforms:
            matched = score_examples(examples, transform)
            cand = result(transform(target), matched / max(len(examples), 1), matched, len(examples), self.name, reason, 0.7)
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best


class EquationTransformSolver(DirectMapSolver):
    name = "EquationTransformSolver"


class AliceSymbolEquationSolver:
    name = "AliceSymbolEquationSolver"

    def solve(self, row) -> dict[str, Any]:
        solvers = [DirectMapSolver(), ReorderSolver(), OperatorRuleSolver(), DeleteRuleSolver()]
        best = None
        for solver in solvers:
            cand = solver.solve(row)
            cand = dict(cand)
            cand["solver_name"] = self.name
            cand["reason"] = f"{solver.name}:{cand.get('reason', '')}"
            if best is None or cand["matched_examples"] > best["matched_examples"]:
                best = cand
        return best


class MultiStepSymbolSolver:
    name = "MultiStepSymbolSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        direct = DirectMapSolver().solve(row)
        mapping_answer = direct["answer"]
        transform = lambda s: "".join({}.get(ch, ch) for ch in s)[::-1]
        matched = 0
        answer = mapping_answer[::-1]
        return result(answer, 0.2 if matched else 0.1, matched, len(examples), self.name, "direct_then_reverse", 0.35)


class DSLSearchSolver(AliceSymbolEquationSolver):
    name = "DSLSearchSolver"


class FallbackModelSolver:
    name = "FallbackModelSolver"

    def solve(self, row) -> dict[str, Any]:
        return result("", 0.0, 0, len(parse_examples(row)), self.name, "disabled_no_model", 0.0)


def all_symbol_solvers():
    return [
        MetadataSymbolSolver(),
        DirectMapSolver(),
        PositionMapSolver(),
        DeleteRuleSolver(),
        KeepOnlySymbolRule(),
        ReorderSolver(),
        PairRuleSolver(),
        OperatorRuleSolver(),
        EquationTransformSolver(),
        AliceSymbolEquationSolver(),
        MultiStepSymbolSolver(),
        DSLSearchSolver(),
        FallbackModelSolver(),
    ]
