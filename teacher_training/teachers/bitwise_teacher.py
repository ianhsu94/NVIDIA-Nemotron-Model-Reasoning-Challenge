from __future__ import annotations

import re

from teacher_training.teacher_base import BaseTeacher, TeacherResult
from teacher_training.utils import parse_examples, target_text


def bits_to_int(bits: str) -> int:
    return int(str(bits).strip(), 2)


def int_to_bits(value: int) -> str:
    return format(value & 255, "08b")


def rol(x: int, k: int) -> int:
    k %= 8
    return ((x << k) | (x >> (8 - k))) & 255


def ror(x: int, k: int) -> int:
    k %= 8
    return ((x >> k) | (x << (8 - k))) & 255


def bit_reverse(x: int) -> int:
    return int(format(x, "08b")[::-1], 2)


def candidates():
    funcs = []
    funcs.append(("identity", lambda x: x))
    funcs.append(("not", lambda x: (~x) & 255))
    funcs.append(("bit_reverse", bit_reverse))
    for k in range(1, 8):
        funcs.extend(
            [
                (f"rol_{k}", lambda x, k=k: rol(x, k)),
                (f"ror_{k}", lambda x, k=k: ror(x, k)),
                (f"shl_{k}", lambda x, k=k: (x << k) & 255),
                (f"shr_{k}", lambda x, k=k: (x >> k) & 255),
                (f"not_rol_{k}", lambda x, k=k: (~rol(x, k)) & 255),
                (f"not_ror_{k}", lambda x, k=k: (~ror(x, k)) & 255),
            ]
        )
    for c in range(256):
        funcs.extend(
            [
                (f"xor_{c}", lambda x, c=c: x ^ c),
                (f"and_{c}", lambda x, c=c: x & c),
                (f"or_{c}", lambda x, c=c: x | c),
                (f"add_{c}", lambda x, c=c: (x + c) & 255),
                (f"sub_{c}", lambda x, c=c: (x - c) & 255),
            ]
        )
    for k in range(1, 8):
        for c in range(256):
            funcs.append((f"rol_{k}_xor_{c}", lambda x, k=k, c=c: rol(x, k) ^ c))
            funcs.append((f"ror_{k}_xor_{c}", lambda x, k=k, c=c: ror(x, k) ^ c))
            funcs.append((f"bit_reverse_xor_{c}", lambda x, c=c: bit_reverse(x) ^ c))
    return funcs


BITWISE_CANDIDATES = candidates()


def solve_gf2(equations: list[tuple[list[int], int]], n_vars: int) -> list[int] | None:
    rows = [coeffs[:] + [rhs] for coeffs, rhs in equations]
    pivot_cols = []
    r = 0
    for c in range(n_vars):
        pivot = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        pivot_cols.append(c)
        r += 1
    for row in rows:
        if not any(row[:n_vars]) and row[-1]:
            return None
    solution = [0] * n_vars
    for row, col in zip(rows, pivot_cols):
        solution[col] = row[-1]
    return solution


def affine_solution(examples: list[tuple[int, int]]):
    if not examples:
        return None
    coeffs_by_bit = []
    for out_bit in range(8):
        equations = []
        for x, y in examples:
            bits = [(x >> bit) & 1 for bit in range(8)] + [1]
            rhs = (y >> out_bit) & 1
            equations.append((bits, rhs))
        sol = solve_gf2(equations, 9)
        if sol is None:
            return None
        coeffs_by_bit.append(sol)

    def apply(x: int) -> int:
        y = 0
        x_bits = [(x >> bit) & 1 for bit in range(8)] + [1]
        for out_bit, coeffs in enumerate(coeffs_by_bit):
            value = 0
            for coeff, bit_value in zip(coeffs, x_bits):
                value ^= coeff & bit_value
            y |= value << out_bit
        return y

    if all(apply(x) == y for x, y in examples):
        return apply
    return None


def base_transforms():
    funcs = [("x", lambda x: x), ("not_x", lambda x: (~x) & 255), ("bit_reverse", bit_reverse)]
    for k in range(1, 8):
        funcs.extend(
            [
                (f"rol_{k}", lambda x, k=k: rol(x, k)),
                (f"ror_{k}", lambda x, k=k: ror(x, k)),
                (f"shl_{k}", lambda x, k=k: (x << k) & 255),
                (f"shr_{k}", lambda x, k=k: (x >> k) & 255),
                (f"not_rol_{k}", lambda x, k=k: (~rol(x, k)) & 255),
                (f"not_ror_{k}", lambda x, k=k: (~ror(x, k)) & 255),
            ]
        )
    return funcs


BASE_TRANSFORMS = base_transforms()


def pairwise_expression_solution(examples: list[tuple[int, int]]):
    if not examples:
        return None, "", 0
    best = (None, "", -1)
    ops = [
        ("and", lambda a, b: a & b),
        ("or", lambda a, b: a | b),
        ("xor", lambda a, b: a ^ b),
    ]
    first_x, first_y = examples[0]
    for name_a, fa in BASE_TRANSFORMS:
        vals_a = [fa(x) for x, _ in examples]
        for name_b, fb in BASE_TRANSFORMS:
            vals_b = [fb(x) for x, _ in examples]
            for op_name, op in ops:
                vals = [op(a, b) & 255 for a, b in zip(vals_a, vals_b)]
                matched = sum(1 for value, (_, y) in zip(vals, examples) if value == y)
                if matched > best[2]:
                    best = (lambda x, fa=fa, fb=fb, op=op: op(fa(x), fb(x)) & 255, f"{name_a}_{op_name}_{name_b}", matched)
                c = vals[0] ^ first_y
                matched_xor = sum(1 for value, (_, y) in zip(vals, examples) if (value ^ c) == y)
                if matched_xor > best[2]:
                    best = (
                        lambda x, fa=fa, fb=fb, op=op, c=c: (op(fa(x), fb(x)) ^ c) & 255,
                        f"({name_a}_{op_name}_{name_b})_xor_{c}",
                        matched_xor,
                    )
                if matched_xor == len(examples):
                    return best
    return best


class BitwiseTeacher(BaseTeacher):
    name = "Teacher-Bitwise"
    category = "bitwise"

    def predict_row(self, row) -> TeacherResult:
        raw_examples = [
            (a.strip(), b.strip())
            for a, b in parse_examples(row)
            if re.fullmatch(r"[01]+", a.strip()) and re.fullmatch(r"[01]+", b.strip())
        ]
        examples = [(bits_to_int(a), bits_to_int(b)) for a, b in raw_examples]
        target = bits_to_int(target_text(row))
        total = len(examples)
        pair_func, pair_reason, pair_matched = pairwise_expression_solution(examples)
        if pair_func is not None and pair_matched == total and total:
            return TeacherResult(int_to_bits(pair_func(target)), 1.0, self.name, pair_matched, total, pair_reason)
        affine = None
        if affine is not None:
            return TeacherResult(int_to_bits(affine(target)), 0.85, self.name, total, total, "affine_gf2")
        best = ("fallback", lambda x: x, -1)
        for name, func in BITWISE_CANDIDATES:
            matched = sum(1 for x, y in examples if func(x) == y)
            if matched > best[2]:
                best = (name, func, matched)
            if matched == total and total:
                answer = int_to_bits(func(target))
                return TeacherResult(answer, 1.0, self.name, matched, total, name)
        if pair_func is not None and pair_matched > best[2]:
            best = (pair_reason, pair_func, pair_matched)
        answer = int_to_bits(best[1](target))
        confidence = best[2] / total if total else 0.0
        return TeacherResult(answer, confidence, self.name, max(best[2], 0), total, best[0])
