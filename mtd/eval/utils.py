import json
from pathlib import Path
from typing import Any, Iterable, List, TypeVar

from tqdm.auto import tqdm

from mtd.utils import PATH_TYPE

T = TypeVar("T")


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f"Loading cache from {file}"):
                yield json.loads(line)


def load_jsonl(file: PATH_TYPE) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def delete_nan(sentences: list):
    cleaned_sentences = []
    for sentence in sentences:
        if isinstance(sentence, str):
            cleaned_sentences.append(sentence)
    num = len(sentences) - len(cleaned_sentences)
    print(f"Deleted {num} NaN values")
    return cleaned_sentences


def compute_average_score(scores: list):
    average_score = sum(scores) / len(scores)
    return average_score


# Before Feb 14, the all experiments were conducted with threshold=0.8
# Since Feb 14, the threshold was changed to 0.2
def compute_ratio(scores: list, threshold=0.2):
    ratio = sum([1 for score in scores if score > threshold]) / len(scores)
    return ratio


def get_eval_project_name(
    model_dir: PATH_TYPE, toxicity_name: str, prompt_name: str
) -> str:
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    toxicity_suffix = f"-{toxicity_name}-{prompt_name}-eval"
    model_name = str(model_dir.parent.name)
    experiment_name = str(model_dir.name)
    project_name = f"{model_name}-{experiment_name}"

    if "-train" in project_name:
        project_name = project_name.replace("-train", toxicity_suffix)
    else:
        project_name = project_name + toxicity_suffix

    return project_name
