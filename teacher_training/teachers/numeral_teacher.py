from __future__ import annotations

import re

from teacher_training.teacher_base import BaseTeacher, TeacherResult
from teacher_training.utils import parse_examples, target_text


ROMAN_MAP = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def int_to_roman(n: int) -> str:
    out = []
    for value, symbol in ROMAN_MAP:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def roman_to_int(text: str) -> int:
    i = 0
    total = 0
    lookup = dict(ROMAN_MAP)
    while i < len(text):
        if i + 1 < len(text) and text[i : i + 2] in lookup:
            total += lookup[text[i : i + 2]]
            i += 2
        else:
            total += lookup[text[i]]
            i += 1
    return total


class NumeralTeacher(BaseTeacher):
    name = "Teacher-Numeral"
    category = "numeral"

    def predict_row(self, row) -> TeacherResult:
        target = target_text(row)
        if re.fullmatch(r"\d+", target):
            answer = int_to_roman(int(target))
            return TeacherResult(answer, 1.0, self.name, reason="decimal_to_roman")
        roman = re.sub(r"[^IVXLCDM]", "", target.upper())
        if roman:
            return TeacherResult(str(roman_to_int(roman)), 1.0, self.name, reason="roman_to_decimal")
        examples = parse_examples(row)
        return TeacherResult(examples[-1][1] if examples else "", 0.0, self.name, len(examples), len(examples), "fallback")

