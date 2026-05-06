from llm4rec.trainers.lora_sft import _split_sft_text, _tokenize_sft


class TinyTokenizer:
    pad_token_id = 0
    eos_token = "|"

    def __call__(self, text, truncation=False, max_length=None, padding=False):
        tokens = [ord(char) for char in text]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        attention = [1] * len(tokens)
        if padding == "max_length" and max_length is not None:
            pad_length = max_length - len(tokens)
            if pad_length > 0:
                tokens = tokens + [self.pad_token_id] * pad_length
                attention = attention + [0] * pad_length
        return {"attention_mask": attention, "input_ids": tokens}


def test_tokenize_sft_masks_prompt_tokens_and_padding():
    row = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "rank candidates"},
            {"role": "assistant", "content": '{"ranked_item_ids": ["i1", "i2"]}'},
        ]
    }

    encoded = _tokenize_sft(row, TinyTokenizer(), max_seq_length=128)
    prefix, _ = _split_sft_text(row)
    prefix_length = len(prefix)

    assert all(label == -100 for label in encoded["labels"][:prefix_length])
    assert encoded["labels"][prefix_length] != -100
    assert encoded["input_ids"][prefix_length] == ord("{")
    assert encoded["labels"][-1] == -100
