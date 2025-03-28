import sys
sys.path.append('/Users/jedtiotuico/python/llama.cpp/llama-cpp-python-scripts')

from transformers import TextStreamer
from unsloth.unsloth.models import FastMTLQwen2Model
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

if __name__ == "__main__":
    model, tokenizer = FastMTLQwen2Model.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=None,
    )

    model.to(device)
    input_ids = tokenizer(
        [
            alpaca_prompt.format(
                "Classify the following into animals, plants, and minerals",  # instruction
                "Oak tree, copper ore, elephant",  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to(device)

    torch.compile(model)
    with torch.no_grad():
        streamer = TextStreamer(tokenizer)

        _ = model.generate(
            input_ids.input_ids,
            use_cache=False,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=128,
        )
