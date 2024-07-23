from transformers import AutoTokenizer

from mtd.utils.config import read_model_config


def load_tokenizer(
    model_path, padding="longest", padding_side="left", pad_token_id=50256, **kwargs
):
    tokenizer_kwargs = {
        "padding": padding,
        "padding_side": padding_side,
        "pad_token_id": pad_token_id,
        **kwargs,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except OSError:
        print("Tokenizer not found, using config file to load tokenizer")
        config = read_model_config(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            config["_name_or_path"], **tokenizer_kwargs
        )

    return tokenizer
