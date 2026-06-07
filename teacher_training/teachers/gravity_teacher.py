from __future__ import annotations

import re
import statistics

from teacher_training.teacher_base import BaseTeacher, TeacherResult
from teacher_training.utils import format_number, parse_examples, target_text


def extract_float(text: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No float in {text!r}")
    return float(m.group(0))


class GravityTeacher(BaseTeacher):
    name = "Teacher-Gravity"
    category = "gravity"

    def predict_row(self, row) -> TeacherResult:
        examples = parse_examples(row)
        g_values = []
        rounded_pairs = []
        for left, right in examples:
            try:
                t = extract_float(left)
                d = extract_float(right)
                rounded_pairs.append((t, d))
                if t:
                    g_values.append(2 * d / (t * t))
            except Exception:
                continue
        candidates = []
        for gi in range(500, 1501):
            g_try = gi / 100
            matched = sum(f"{0.5 * g_try * t * t:.2f}" == f"{d:.2f}" for t, d in rounded_pairs)
            if matched == len(rounded_pairs) and rounded_pairs:
                candidates.append(g_try)
        if candidates:
            g = statistics.median(candidates)
        else:
            xs = [0.5 * t * t for t, _ in rounded_pairs]
            ys = [d for _, d in rounded_pairs]
            g = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs) if xs else 9.8
        t = extract_float(target_text(row))
        answer = f"{0.5 * g * t * t:.2f}"
        return TeacherResult(answer, 1.0 if g_values else 0.5, self.name, len(g_values), len(examples), f"g={g:.6f}")
