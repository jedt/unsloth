import torch
import sys
sys.path.append('/Users/jedtiotuico/python/llama.cpp/llama-cpp-python-scripts')

from unsloth.unsloth.models import FastMTLQwen2Model, FastMTLQwen2Attention_forward, LLamaForCausalLM_forward
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu")

def modified_forward(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states

def benchmark(model, inputs, num_runs=100):
    for _ in range(10):
        _ = model(**inputs)

    # Timing with MPS events for accuracy
    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    torch.mps.synchronize()
    start_event.record()
    for _ in range(num_runs):
        _ = model(**inputs)
    end_event.record()
    torch.mps.synchronize()

    return start_event.elapsed_time(end_event) / num_runs

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = FastMTLQwen2Model.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=None,
    )
    model.eval()  # Disable dropout for consistent timing

    # Prepare input (using a small sequence for simplicity)
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Benchmark original forward pass
    original_time = benchmark(model, inputs)
    print(f"Original forward pass avg time: {original_time:.3f} ms")

    # Monkey-patch to use optimized forward method
    # original_forward = LlamaRMSNorm.forward
    # LlamaRMSNorm.forward = modified_forward
    Qwen2Attention      .forward = FastMTLQwen2Attention_forward
    Qwen2ForCausalLM    .forward = LLamaForCausalLM_forward

    # Benchmark optimized forward pass
    modified_time = benchmark(model, inputs)
    print(f"Optimized forward pass avg time: {modified_time:.3f} ms")

    # Calculate and print improvement
    improvement = (original_time - modified_time) / original_time * 100
    print(f"Performance improvement: {improvement:.2f}%")
