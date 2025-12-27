import numpy as np

from sentiment_span_extractor.metrics.jaccard import calculate_jaccard_score, jaccard


def test_jaccard_identical():
    assert jaccard("hello world", "hello world") == 1.0


def test_jaccard_no_overlap():
    assert jaccard("hello world", "foo bar") == 0.0


def test_jaccard_partial_overlap():
    result = jaccard("hello world", "hello foo")
    assert 0.0 < result < 1.0
    # intersection: {"hello"} = 1, union: {"hello", "world", "foo"} = 3
    # J = 1 / 3 = 0.333...
    assert abs(result - 1.0 / 3.0) < 0.01


def test_jaccard_empty_strings():
    assert jaccard("", "") == 1.0


def test_calculate_jaccard_score():
    original_tweet = "I am happy"
    target_string = "happy"
    sentiment_val = "positive"
    idx_start = 2
    idx_end = 2
    offsets = np.array([[0, 1], [2, 4], [5, 10]])

    score, output = calculate_jaccard_score(
        original_tweet=original_tweet,
        target_string=target_string,
        sentiment_val=sentiment_val,
        idx_start=idx_start,
        idx_end=idx_end,
        offsets=offsets,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(output, str)


def test_calculate_jaccard_score_neutral():
    original_tweet = "I am neutral"
    target_string = "neutral"
    sentiment_val = "neutral"
    idx_start = 0
    idx_end = 0
    offsets = np.array([[0, 1], [2, 4], [5, 12]])

    score, output = calculate_jaccard_score(
        original_tweet=original_tweet,
        target_string=target_string,
        sentiment_val=sentiment_val,
        idx_start=idx_start,
        idx_end=idx_end,
        offsets=offsets,
    )

    assert output == original_tweet
