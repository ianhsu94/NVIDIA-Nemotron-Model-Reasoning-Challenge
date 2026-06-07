from __future__ import annotations


def route_prompt(prompt: str) -> str:
    p = prompt.lower()
    if any(x in p for x in ["roman", "numeral", "write the number", "decimal"]):
        return "numeral"
    if any(x in p for x in ["fall", "gravity", "height", " g ", " t =", "0.5*g*t^2"]):
        return "gravity"
    if any(x in p for x in ["convert", " km", " cm", " kg", " hour", " measurement", "unit"]):
        return "unit"
    if any(x in p for x in ["and", "or", "xor", "not", "binary", "bit manipulation"]):
        return "bitwise"
    if any(x in p for x in ["cipher", "decrypt", "encrypt", "encoded", "decoded", "encryption"]):
        return "cipher"
    if any(x in p for x in ["transformation rules", "wonderland", "equation", "determine the result for"]):
        return "symbol"
    return "symbol"

