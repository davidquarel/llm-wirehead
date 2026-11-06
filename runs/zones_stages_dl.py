# Download the Llama-3.1-8B training-stage checkpoints for the zones stage comparison (run on arena9-trouble).
from huggingface_hub import snapshot_download as s
for m in ["unsloth/Meta-Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B"]:
    print(m, s(m, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]), flush=True)
