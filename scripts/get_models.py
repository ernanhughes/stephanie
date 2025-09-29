from huggingface_hub import snapshot_download

# Replace with your Hugging Face token if required (for gated models)
# export HUGGINGFACE_HUB_TOKEN="your_token_here"

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# This will download the whole repo snapshot into ~/.cache/huggingface/hub
local_dir = snapshot_download(repo_id=model_id)

print("Model files are in:", local_dir)
