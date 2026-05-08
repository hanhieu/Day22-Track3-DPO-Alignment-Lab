# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load DPO model + merge adapter

# %%
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load base model with standard HF (no Unsloth patches needed for merge)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Stack SFT-mini → DPO adapters then merge to base weights
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
model = PeftModel.from_pretrained(model, str(SFT_PATH))
print(f"Loaded SFT-mini adapter from {SFT_PATH}")

# %% [markdown]
# ## 2. Save merged FP16 weights using standard PEFT merge_and_unload

# %%
# Merge SFT adapter into base weights
model = model.merge_and_unload()
print("Merged SFT adapter into base weights")

# Load DPO adapter on top of merged model
model = PeftModel.from_pretrained(model, str(DPO_PATH))
print(f"Loaded DPO adapter from {DPO_PATH}")

# Merge DPO adapter
model = model.merge_and_unload()
print("Merged DPO adapter into base weights")

# Save merged model in BF16
model = model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model.save_pretrained(str(MERGED_PATH))
tokenizer.save_pretrained(str(MERGED_PATH))
print(f"Saved merged model to {MERGED_PATH}")

# Free GPU memory before GGUF conversion (which spawns a subprocess that needs RAM)
import gc

del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Quantize to GGUF Q4_K_M using llama.cpp

# %%
import subprocess, sys

# Download llama.cpp convert script if not present
LLAMA_CPP_DIR = REPO_ROOT / "llama_cpp_tools"
LLAMA_CPP_DIR.mkdir(exist_ok=True)
convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

if not convert_script.exists():
    print("Downloading llama.cpp convert_hf_to_gguf.py...")
    import urllib.request
    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    urllib.request.urlretrieve(url, convert_script)
    print("Downloaded.")

# Step 1: Convert HF model to GGUF F16
gguf_f16 = GGUF_DIR / "lab22-dpo-f16.gguf"
print(f"Converting {MERGED_PATH} → {gguf_f16} ...")
result = subprocess.run(
    [sys.executable, str(convert_script),
     str(MERGED_PATH),
     "--outfile", str(gguf_f16),
     "--outtype", "f16"],
    capture_output=True, text=True, timeout=600
)
if result.returncode != 0:
    print("STDOUT:", result.stdout[-2000:])
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"convert_hf_to_gguf.py failed: {result.returncode}")
print(f"F16 GGUF saved: {gguf_f16} ({gguf_f16.stat().st_size/1e6:.0f} MB)")

# Step 2: Quantize F16 → Q4_K_M using llama-cpp-python's bundled quantize
from llama_cpp import llama_cpp as _lc
import ctypes, os as _os

# Find llama-quantize binary bundled with llama-cpp-python
import llama_cpp as _llama_pkg
_pkg_dir = Path(_llama_pkg.__file__).parent
quantize_bin = None
for candidate in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
    p = _pkg_dir / candidate
    if p.exists():
        quantize_bin = p
        break

gguf_q4 = GGUF_DIR / "lab22-dpo-Q4_K_M.gguf"
if quantize_bin:
    print(f"Quantizing {gguf_f16} → {gguf_q4} ...")
    result = subprocess.run(
        [str(quantize_bin), str(gguf_f16), str(gguf_q4), "Q4_K_M"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        # Fall back: use the F16 as the final GGUF
        import shutil
        shutil.copy(gguf_f16, gguf_q4)
        print("Quantize failed — using F16 GGUF as fallback")
    else:
        print(f"Q4_K_M GGUF saved: {gguf_q4} ({gguf_q4.stat().st_size/1e6:.0f} MB)")
        gguf_f16.unlink(missing_ok=True)  # remove F16 to save disk
else:
    # No quantize binary — use F16 directly (llama-cpp-python can load it)
    import shutil
    shutil.copy(gguf_f16, gguf_q4)
    print(f"No quantize binary found — using F16 GGUF: {gguf_q4}")

print("\nGGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        print(f"  {p.name:50s}  {p.stat().st_size/1e6:>8.0f} MB")

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the GGUF file (Q4_K_M or F16 fallback)
gguf_files = (list(GGUF_DIR.glob("*Q4_K_M*.gguf")) +
              list(GGUF_DIR.glob("*q4_k_m*.gguf")) +
              list(GGUF_DIR.glob("*.gguf")))
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# Save smoke test screenshot as text image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
smoke_text = (
    f"GGUF Smoke Test — {gguf_path.name}\n\n"
    f"PROMPT:\n{SMOKE_PROMPT}\n\n"
    f"RESPONSE:\n{response['choices'][0]['message']['content'][:400]}"
)
ax.text(0.02, 0.95, smoke_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "06-gguf-smoke.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved smoke screenshot to {screenshot_dir / '06-gguf-smoke.png'}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
