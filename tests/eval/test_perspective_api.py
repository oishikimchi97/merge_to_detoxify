from pathlib import Path

from googleapiclient.errors import HttpError

from mtd.eval.perspective_api import PerspectiveAPI, PerspectiveWorker


def test_perspective_api():
    api = PerspectiveAPI()

    text_success = "Testing"
    text_error = "x" * (20480 + 1)

    score_1, error_1 = api.request(text_success)[0]
    assert score_1 and not error_1

    score_2, error_2 = api.request(text_error)[0]
    assert not score_2 and isinstance(error_2, HttpError)

    # multi_score, multi_error = zip(*api.request([text_success, text_error]))
    # assert multi_score == (score_1, score_2)
    # assert tuple(map(str, multi_error)) == tuple(map(str, (error_1, error_2)))


def test_perspective_worker():
    perspective_file = Path("logs/perspective_test.jsonl")
    test_generation_iter = ["Testing"] * 100
    perspective = PerspectiveWorker(
        out_file=perspective_file,
        total=len(test_generation_iter),
        rate_limit=10,
    )

    # Generate and collate perspective scores
    generations = []
    for i, gen in enumerate(test_generation_iter):
        generations.append(gen)
        perspective(f"generation-{i}", gen)
    perspective.stop()
