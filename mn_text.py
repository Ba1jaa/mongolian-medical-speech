"""
Mongolian (Cyrillic) text normalization for ASR training and TTS input.

- lowercases
- expands numbers to Mongolian words  (25 -> "хорин тав")
- expands common medical units        (5 мг -> "тав миллиграмм")
- removes punctuation / Latin characters

IMPORTANT: have a native Mongolian speaker review the number rules and the
abbreviation list below. Extend ABBREVIATIONS with your own medical terms.
"""
import re

ONES = ["", "нэг", "хоёр", "гурав", "дөрөв", "тав", "зургаа", "долоо", "найм", "ес"]
TENS_STAND = {1: "арав", 2: "хорь", 3: "гуч", 4: "дөч", 5: "тавь",
              6: "жар", 7: "дал", 8: "наян", 9: "ер"}
TENS_COMB = {1: "арван", 2: "хорин", 3: "гучин", 4: "дөчин", 5: "тавин",
             6: "жаран", 7: "далан", 8: "наян", 9: "ерэн"}

UNITS = {
    "мкг": "микрограмм", "мг": "миллиграмм", "мл": "миллилитр",
    "кг": "килограмм", "мм": "миллиметр", "см": "сантиметр",
    "г": "грамм", "л": "литр",
}

# Add your own medical abbreviations here (lowercase key -> spoken form)
ABBREVIATIONS = {
    "жишээ нь": "жишээ нь",
    "т.е": "тухайн нь",
    "мэс.": "мэс засал",
}


def _below_thousand(n: int, final: bool):
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append(ONES[h])
        parts.append("зуу" if (r == 0 and final) else "зуун")
    t, o = divmod(r, 10)
    if t:
        parts.append(TENS_STAND[t] if (o == 0 and final) else TENS_COMB[t])
    if o:
        parts.append(ONES[o])
    return parts


def number_to_words(n: int) -> str:
    if n == 0:
        return "тэг"
    if n < 0:
        return "хасах " + number_to_words(-n)
    if n >= 10 ** 12:
        return " ".join(ONES[int(d)] or "тэг" for d in str(n))  # read digit by digit
    parts = []
    for unit, name in ((10 ** 9, "тэрбум"), (10 ** 6, "сая"), (10 ** 3, "мянга")):
        q, n = divmod(n, unit)
        if q:
            if not (q == 1 and unit == 1000):
                parts += _below_thousand(q, final=False)
            parts.append(name)
    if n:
        parts += _below_thousand(n, final=True)
    return " ".join(parts)


def _expand_decimal(m):
    whole, frac = m.group(1), m.group(2)
    return f"{number_to_words(int(whole))} таслал " + " ".join(
        number_to_words(int(d)) for d in frac)


def _expand_number(m):
    return number_to_words(int(m.group(0)))


def normalize_text(text: str, keep_punct: bool = False) -> str:
    t = text.lower().strip()

    for k, v in ABBREVIATIONS.items():
        t = t.replace(k, v)

    # percent, degrees
    t = re.sub(r"%", " хувь ", t)
    t = re.sub(r"°\s*c", " градус цельс ", t)

    # number + unit: "5 мг", "2.5мл"  -> expand unit first (keeps the number for later)
    unit_re = "|".join(sorted(UNITS, key=len, reverse=True))
    t = re.sub(rf"(\d+(?:[.,]\d+)?)\s*({unit_re})(?!\w)",
               lambda m: f"{m.group(1)} {UNITS[m.group(2)]}", t)

    # decimals, then integers
    t = re.sub(r"(\d+)[.,](\d+)", _expand_decimal, t)
    t = re.sub(r"\d+", _expand_number, t)

    if keep_punct:
        t = re.sub(r"[^а-яёөү\s.,?!-]", " ", t)
    else:
        t = re.sub(r"[^а-яёөү\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    for s in ["Өдөрт 2 удаа 5 мг уух", "Тун: 2.5 мл", "123 өвчтөн", "2026 он", "20 хүн, 100 тун"]:
        print(s, "->", normalize_text(s))
