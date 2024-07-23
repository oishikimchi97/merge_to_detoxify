import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import datasets
from mtd.utils.transformer import load_tokenizer


def get_context_length(model):
    if hasattr(model.config, "n_positions"):
        context_length = model.config.n_positions
    elif hasattr(model.config, "max_position_embeddings"):
        context_length = model.config.max_position_embeddings
    else:
        raise ValueError(
            "Model should have either `n_positions` or `max_position_embeddings` attribute"
        )

    return context_length


def chunk_token(input_ids, max_length=1024, stride=512, pad_token_id=50256):
    seq_len = input_ids.size(0)
    chunks = []
    targets = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        target_len = end_loc - prev_end_loc  # may be different from stride on last loop
        chunked_ids = input_ids[begin_loc:end_loc]
        chunked_len = len(chunked_ids)
        if chunked_len < max_length:
            # Add padding
            pad_length = max_length - chunked_len
            chunked_ids = torch.cat(
                [
                    chunked_ids,
                    torch.ones(pad_length, dtype=chunked_ids.dtype) * pad_token_id,
                ],
                dim=0,
            )
            # Modify target_len to include padding
            target_len += pad_length
        target_ids = chunked_ids.clone()
        # Replace the input_ids with mask tokens
        # because do not calculate perplexity on the overlapped tokens including pad tokens
        target_ids[:-target_len] = -100

        chunks.append(chunked_ids)
        targets.append(target_ids)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return {"input_ids": chunks, "labels": targets}


def validate_perplexity(
    model_path,
    dataset_path="wikitext",
    dataset_name="wikitext-103-v1",
    dataset_split="test",
    batch_size=4,
    stride=512,
):
    wikitext_dataset = datasets.load_dataset(
        dataset_path, dataset_name, split=dataset_split
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.config.use_cache = True
    tokenizer = load_tokenizer(model_path)
    tokenizer.pad_token_id = (
        model.config.eos_token_id if model.config.eos_token_id is not None else 50256
    )

    encodings = tokenizer(
        "\n\n".join(wikitext_dataset["text"]),
        return_tensors="pt",
    )
    max_length = get_context_length(model)
    # max_length = 1024
    eos_token_id = (
        model.config.eos_token_id if model.config.eos_token_id is not None else 50256
    )

    chunked_dict = chunk_token(
        encodings["input_ids"][0], max_length, stride=stride, pad_token_id=eos_token_id
    )
    dataset = datasets.Dataset.from_dict(chunked_dict)
    dataset.set_format(type="torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    nlls = []
    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            input_ids, labels = batch["input_ids"], batch["labels"]
            input_ids, labels = input_ids.to(model.device), labels.to(model.device)
            outputs = model(input_ids, labels=labels)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.tensor(nlls)).mean()

    print(f"Perplexity: {ppl}")
    return ppl
