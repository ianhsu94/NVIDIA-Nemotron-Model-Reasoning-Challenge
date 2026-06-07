from __future__ import annotations

import re
import statistics

from teacher_training.teacher_base import BaseTeacher, TeacherResult
from teacher_training.utils import format_number, parse_examples, target_text


def first_float(text: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"No float in {text!r}")
    return float(m.group(0))


class UnitTeacher(BaseTeacher):
    name = "Teacher-Unit"
    category = "unit"

    def predict_row(self, row) -> TeacherResult:
        examples = parse_examples(row)
        ratios = []
        rounded_pairs = []
        for left, right in examples:
            try:
                x = first_float(left)
                y = first_float(right)
                rounded_pairs.append((x, y))
                if x:
                    ratios.append(y / x)
            except Exception:
                continue
        candidates = []
        for ri in range(2000, 30001):
            ratio_try = ri / 10000
            matched = sum(f"{x * ratio_try:.2f}" == f"{y:.2f}" for x, y in rounded_pairs)
            if matched == len(rounded_pairs) and rounded_pairs:
                candidates.append(ratio_try)
        if candidates:
            ratio = statistics.median(candidates)
        else:
            ratio = statistics.median(ratios) if ratios else 1.0
        value = first_float(target_text(row))
        answer = f"{value * ratio:.2f}"
        return TeacherResult(answer, 1.0 if ratios else 0.4, self.name, len(ratios), len(examples), f"ratio={ratio:.8f}")
