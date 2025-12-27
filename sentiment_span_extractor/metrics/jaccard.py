
import numpy as np


def jaccard(str1: str, str2: str) -> float:
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard_score(
    original_tweet: str,
    target_string: str,
    sentiment_val: str,
    idx_start: int,
    idx_end: int,
    offsets: np.ndarray,
    verbose: bool = False,
    min_words_for_extraction: int = 2,
) -> tuple[float, str]:
    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        if ix < len(offsets):
            offset_start, offset_end = offsets[ix]
            filtered_output += original_tweet[offset_start:offset_end]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < min_words_for_extraction:
        filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output
