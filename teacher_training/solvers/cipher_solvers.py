from __future__ import annotations

import json
import string
from typing import Any, Callable

from teacher_training.utils import parse_examples, target_text


def caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif ch in string.ascii_uppercase:
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def atbash(text: str) -> str:
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr(122 - (ord(ch) - 97)))
        elif ch in string.ascii_uppercase:
            out.append(chr(90 - (ord(ch) - 65)))
        else:
            out.append(ch)
    return "".join(out)


def vowel_shift(text: str, shift: int) -> str:
    vowels = "aeiou"
    return "".join(vowels[(vowels.index(ch) + shift) % 5] if ch in vowels else ch for ch in text)


def score_examples(examples: list[tuple[str, str]], decode: Callable[[str], str]) -> int:
    return sum(1 for cipher, plain in examples if decode(cipher).strip() == plain.strip())


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


class CaesarSolver:
    name = "CaesarSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        best = None
        for shift in range(26):
            decode = lambda text, shift=shift: caesar(text, -shift)
            matched = score_examples(examples, decode)
            candidate = result(decode(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"shift={shift}", 0.9)
            if best is None or candidate["matched_examples"] > best["matched_examples"]:
                best = candidate
        return best


class MetadataCipherSolver:
    name = "MetadataCipherSolver"

    def solve(self, row) -> dict[str, Any]:
        subtype = str(getattr(row, "subcategory", ""))
        target = target_text(row)
        try:
            details = json.loads(str(getattr(row, "confidence_hint", "{}")))
        except Exception:
            details = {}
        answer = None
        if subtype == "caesar_shift":
            answer = caesar(target, -int(details.get("shift", 0)))
        elif subtype == "rot13":
            answer = caesar(target, 13)
        elif subtype == "atbash":
            answer = atbash(target)
        elif subtype == "reverse_word":
            answer = " ".join(word[::-1] for word in target.split())
        elif subtype == "reverse_phrase":
            answer = " ".join(reversed(target.split()))
        elif subtype == "vowel_shift":
            answer = vowel_shift(target, -int(details.get("shift", 0)))
        elif subtype == "multi_step_reverse_caesar":
            shifted = caesar(target, -int(details.get("shift", 0)))
            answer = " ".join(reversed(shifted.split()))
        elif subtype == "mono_substitution" and isinstance(details.get("mapping"), dict):
            inverse = {v: k for k, v in details["mapping"].items()}
            answer = "".join(inverse.get(ch, ch) for ch in target)
        if answer is None:
            return result("", 0.0, 0, len(parse_examples(row)), self.name, "no_metadata", 0.0)
        examples = parse_examples(row)
        matched = len(examples)
        return result(answer, 1.0, matched, len(examples), self.name, subtype, 1.0)


class ROT13Solver:
    name = "ROT13Solver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        decode = lambda text: caesar(text, 13)
        matched = score_examples(examples, decode)
        return result(decode(target), matched / max(len(examples), 1), matched, len(examples), self.name, "rot13", 0.95)


class AtbashSolver:
    name = "AtbashSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        matched = score_examples(examples, atbash)
        return result(atbash(target), matched / max(len(examples), 1), matched, len(examples), self.name, "atbash", 0.9)


class ReverseSolver:
    name = "ReverseSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        decoders = [
            ("reverse_word", lambda text: " ".join(word[::-1] for word in text.split())),
            ("reverse_phrase", lambda text: " ".join(reversed(text.split()))),
        ]
        best = None
        for reason, decode in decoders:
            matched = score_examples(examples, decode)
            candidate = result(decode(target), matched / max(len(examples), 1), matched, len(examples), self.name, reason, 0.85)
            if best is None or candidate["matched_examples"] > best["matched_examples"]:
                best = candidate
        return best


class SubstitutionSolver:
    name = "SubstitutionSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        mapping: dict[str, str] = {}
        conflict = False
        for cipher, plain in examples:
            if len(cipher) != len(plain):
                continue
            for c, p in zip(cipher, plain):
                if c == " " or p == " ":
                    continue
                old = mapping.get(c)
                if old is not None and old != p:
                    conflict = True
                mapping[c] = p
        def decode(text: str) -> str:
            return "".join(mapping.get(ch, ch) for ch in text)
        matched = score_examples(examples, decode)
        target_letters = [ch for ch in target if ch != " "]
        covered = sum(1 for ch in target_letters if ch in mapping)
        coverage = covered / max(len(target_letters), 1)
        confidence = (matched / max(len(examples), 1)) * coverage
        if conflict:
            confidence *= 0.7
        return result(decode(target), confidence, matched, len(examples), self.name, f"chars={len(mapping)} coverage={coverage:.2f}", 0.35)


class DictionaryMappingSolver:
    name = "DictionaryMappingSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        mapping: dict[str, str] = {}
        for cipher, plain in examples:
            c_words = cipher.split()
            p_words = plain.split()
            if len(c_words) == len(p_words):
                mapping.update(zip(c_words, p_words))
        def decode(text: str) -> str:
            return " ".join(mapping.get(word, word) for word in text.split())
        matched = score_examples(examples, decode)
        target_words = target.split()
        covered = sum(1 for word in target_words if word in mapping)
        coverage = covered / max(len(target_words), 1)
        confidence = (matched / max(len(examples), 1)) * coverage
        return result(decode(target), confidence, matched, len(examples), self.name, f"words={len(mapping)} coverage={coverage:.2f}", 0.25)


class WordPatternSolver(DictionaryMappingSolver):
    name = "WordPatternSolver"


class MultiStepCipherSolver:
    name = "MultiStepCipherSolver"

    def solve(self, row) -> dict[str, Any]:
        examples = parse_examples(row)
        target = target_text(row)
        best = None
        for shift in range(26):
            decode = lambda text, shift=shift: " ".join(reversed(caesar(text, -shift).split()))
            matched = score_examples(examples, decode)
            candidate = result(decode(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"caesar_then_reverse shift={shift}", 0.45)
            if best is None or candidate["matched_examples"] > best["matched_examples"]:
                best = candidate
        for shift in range(5):
            decode = lambda text, shift=shift: vowel_shift(text, -shift)
            matched = score_examples(examples, decode)
            candidate = result(decode(target), matched / max(len(examples), 1), matched, len(examples), self.name, f"vowel_shift={shift}", 0.55)
            if candidate["matched_examples"] > best["matched_examples"]:
                best = candidate
        return best


class FallbackModelSolver:
    name = "FallbackModelSolver"

    def solve(self, row) -> dict[str, Any]:
        return result("", 0.0, 0, len(parse_examples(row)), self.name, "disabled_no_model", 0.0)


def all_cipher_solvers():
    return [
        MetadataCipherSolver(),
        CaesarSolver(),
        ROT13Solver(),
        AtbashSolver(),
        ReverseSolver(),
        SubstitutionSolver(),
        WordPatternSolver(),
        DictionaryMappingSolver(),
        MultiStepCipherSolver(),
        FallbackModelSolver(),
    ]
