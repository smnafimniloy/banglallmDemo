"""
banglallm_streamlit.py
══════════════════════
Streamlit UI for testing BanglaLLM text generation.

Usage:
    pip install streamlit
    streamlit run banglallm_streamlit.py -- --models-root ./BanglaLLM

Optional args (pass after --):
    --models-root PATH    Root dir to scan for models (default: ./BanglaLLM)
    --sp-model PATH       Path to bangla_spm.model (auto-detected if omitted)
    --device auto|cuda|cpu
"""

import argparse
import glob
import json
import os
import sys
import time
import gc

import streamlit as st

# ── CLI args (Streamlit passes them after --) ────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--models-root", type=str, default="./BanglaLLM")
parser.add_argument("--sp-model", type=str, default=None)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--hf-repo", type=str, default=None,
    help="HuggingFace repo ID to download model from "
         "(e.g. nafim/banglallm-test). Skipped if models-root exists.",
)
args, _ = parser.parse_known_args()


# ════════════════════════════════════════════════════════════════════
# AUTO-DOWNLOAD FROM HUGGINGFACE (for cloud hosting)
# ════════════════════════════════════════════════════════════════════

def ensure_model_downloaded():
    """Download model from HF Hub if not already on disk."""
    hf_repo = args.hf_repo or os.environ.get("HF_REPO")

    # Streamlit Cloud stores secrets in st.secrets, not os.environ
    if not hf_repo:
        try:
            hf_repo = st.secrets.get("HF_REPO")
        except Exception:
            pass

    if not hf_repo:
        # Last resort: hardcoded fallback
        hf_repo = "smnafimniloy/banglallm-test"

    models_exist = False
    if os.path.exists(args.models_root):
        for _, _, files in os.walk(args.models_root):
            if "config.json" in files:
                models_exist = True
                break

    if models_exist:
        return

    try:
        from huggingface_hub import snapshot_download
        dest = os.path.join(args.models_root, "hf_model")
        os.makedirs(dest, exist_ok=True)
        with st.status(f"Downloading model from {hf_repo}...", expanded=True) as status:
            st.write(f"Repository: {hf_repo}")
            st.write(f"Destination: {dest}")
            snapshot_download(
                repo_id=hf_repo,
                local_dir=dest,
                local_dir_use_symlinks=False,
            )
            # List downloaded files
            for root, dirs, files in os.walk(dest):
                for f in files:
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp) / 1e6
                    st.write(f"  {os.path.relpath(fp, dest)} ({size:.1f} MB)")
            status.update(label="Model downloaded", state="complete")
    except Exception as e:
        st.error(f"Failed to download model from {hf_repo}: {e}")
        st.stop()

ensure_model_downloaded()


# ════════════════════════════════════════════════════════════════════
# BACKEND
# ════════════════════════════════════════════════════════════════════

def find_sp_model(root):
    if args.sp_model and os.path.exists(args.sp_model):
        return args.sp_model
    # Search for any .model file under root
    for pattern in ["**/bangla_spm.model", "**/*.model"]:
        hits = glob.glob(os.path.join(root, pattern), recursive=True)
        if hits:
            return hits[0]
    # Also check current directory
    for pattern in ["**/bangla_spm.model", "**/*.model"]:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    return None


@st.cache_resource
def load_sp():
    import sentencepiece as spm
    path = find_sp_model(args.models_root)
    if path is None:
        st.error(f"No .model file found under {args.models_root}. "
                 "Pass --sp-model explicitly.")
        st.stop()
    return spm.SentencePieceProcessor(model_file=path), path


def pick_device():
    if args.device and args.device != "auto":
        return args.device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def discover_models(root):
    models = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames:
            try:
                with open(os.path.join(dirpath, "config.json")) as f:
                    cfg = json.load(f)
                if "model_type" not in cfg and "architectures" not in cfg:
                    continue
                rel = os.path.relpath(dirpath, root)
                size_mb = sum(
                    os.path.getsize(os.path.join(dirpath, fn)) / 1e6
                    for fn in filenames
                )
                h = cfg.get("hidden_size", 0)
                L = cfg.get("num_hidden_layers", 0)
                V = cfg.get("vocab_size", 0)
                ffn = cfg.get("intermediate_size", 0)
                n_kv = cfg.get("num_key_value_heads",
                               cfg.get("num_attention_heads", 0))
                n_q = cfg.get("num_attention_heads", 0)
                est_params = 0
                if h and L and V:
                    embed = V * h * (1 if cfg.get("tie_word_embeddings") else 2)
                    attn = h * h + n_kv * (h // max(n_q, 1)) * h * 2 + h * h
                    ffn_p = h * ffn * 3 if ffn else h * h * 4 * 3
                    est_params = embed + L * (attn + ffn_p)

                models.append({
                    "path": dirpath,
                    "name": rel,
                    "size_mb": round(size_mb, 1),
                    "model_type": cfg.get("model_type", "unknown"),
                    "hidden_size": h,
                    "num_layers": L,
                    "num_heads": n_q,
                    "num_kv_heads": n_kv,
                    "vocab_size": V,
                    "intermediate_size": ffn,
                    "max_position_embeddings": cfg.get("max_position_embeddings", 0),
                    "est_params_m": round(est_params / 1e6, 1) if est_params else None,
                    "attn_impl": cfg.get("attn_implementation", ""),
                })
            except Exception:
                continue
    models.sort(key=lambda m: m["name"])
    return models


@st.cache_resource
def load_model(model_path):
    import torch
    from transformers import LlamaForCausalLM, AutoModelForCausalLM

    device = pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )

    if device == "cpu":
        model = model.float()
    model.eval()

    actual_params = sum(p.numel() for p in model.parameters())
    return model, device, actual_params


def generate_text(model, device, sp, prompt, max_new_tokens=128,
                  temperature=0.8, top_p=0.9, top_k=50,
                  repetition_penalty=1.2, suppress_eos=False):
    import torch

    ids = [sp.bos_id()] + sp.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=None if suppress_eos else sp.eos_id(),
            pad_token_id=sp.pad_id(),
        )
    elapsed = time.perf_counter() - t0

    new_ids = output[0][len(ids):].tolist()
    generated = sp.decode(new_ids)

    return {
        "generated": generated,
        "prompt_tokens": len(ids),
        "new_tokens": len(new_ids),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": round(len(new_ids) / max(elapsed, 0.001), 1),
    }


# ════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BanglaLLM Playground",
    page_icon="✦",
    layout="wide",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600&display=swap');

/* ── Bengali text rendering ── */
.bengali-text, .prompt-echo, .generated-text {
    font-family: 'Noto Sans Bengali', sans-serif !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}
.prompt-echo {
    color: #8b95a5 !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}
.generated-text {
    color: #e2e8f0 !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

/* ── Base: tighten default Streamlit spacing ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] > div {
    gap: 0.5rem !important;
}

/* ── Sidebar cleanup ── */
section[data-testid="stSidebar"] .block-container {
    padding: 1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    font-size: 13px !important;
}

/* ── Result cards ── */
.result-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    font-family: monospace;
    font-size: 12px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.result-card-header .model-label {
    color: #38bdf8;
    font-weight: 600;
}
.result-card-header .stats-label {
    color: #8b95a5;
}
.result-card-body {
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* ── Quick prompt buttons ── */
.qp-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
}
.qp-grid button {
    font-family: 'Noto Sans Bengali', sans-serif !important;
    font-size: 13px !important;
    padding: 6px 12px !important;
    border-radius: 20px !important;
    white-space: nowrap !important;
}

/* ── Tablet ── */
@media (max-width: 1024px) {
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* ── Mobile ── */
@media (max-width: 768px) {
    /* Title */
    h1 { font-size: 1.3rem !important; }

    /* Main content padding */
    .block-container {
        padding: 0.75rem 0.5rem !important;
    }

    /* Bengali text smaller */
    .prompt-echo, .generated-text {
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* Text area */
    textarea {
        font-size: 14px !important;
        min-height: 80px !important;
    }

    /* Result header stacks */
    .result-card-header {
        flex-direction: column;
        align-items: flex-start;
    }

    /* Warning banner tighter */
    [data-testid="stAlert"] {
        padding: 8px 12px !important;
        font-size: 13px !important;
    }
    [data-testid="stAlert"] p {
        font-size: 13px !important;
    }
}

/* ── Small phones ── */
@media (max-width: 480px) {
    h1 { font-size: 1.1rem !important; }
    .prompt-echo, .generated-text {
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
        font-size: 12px !important;
    }
}

/* ── Touch: bigger tap targets ── */
@media (pointer: coarse) {
    button[kind="primary"], button[kind="secondary"],
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        min-height: 44px !important;
        font-size: 14px !important;
    }
    [data-testid="stSlider"] > div {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }
    [data-testid="stCheckbox"] label {
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("✦ বাংলাLLM Playground")
st.warning(
    "⚠️ **This model is in early development.** "
    "Output is largely random and not meaningful. "
    "The model has not been fully trained yet — "
    "this playground is for testing and research purposes only.",
    icon="🚧",
)

# ── Load SP ──
sp, sp_path = load_sp()

# ── Discover models ──
models = discover_models(args.models_root)

if not models:
    st.error(f"No models found under `{args.models_root}`. "
             "Train a model first, then re-run.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**Device:** `{pick_device()}`")
    st.markdown(f"**Tokenizer:** `{os.path.basename(sp_path)}` "
                f"({sp.get_piece_size():,} vocab)")
    st.divider()

    # Model selector
    st.subheader("Model")
    model_names = [m["name"] for m in models]
    selected_idx = st.selectbox(
        "Select model",
        range(len(models)),
        format_func=lambda i: f"{models[i]['name']} ({models[i]['size_mb']} MB)",
        label_visibility="collapsed",
    )
    selected_model = models[selected_idx]

    # Model info card
    with st.expander("Model details", expanded=True):
        m = selected_model
        param_str = "—"
        if m["est_params_m"]:
            param_str = (f"{m['est_params_m']/1000:.2f}B"
                         if m["est_params_m"] >= 1000
                         else f"{m['est_params_m']}M")
        st.markdown(f"""
| | |
|---|---|
| **Type** | `{m['model_type']}` |
| **Params (est.)** | `{param_str}` |
| **Layers** | `{m['num_layers']}` |
| **Hidden** | `{m['hidden_size']}` |
| **FFN** | `{m['intermediate_size']}` |
| **Heads** | `{m['num_heads']}Q / {m['num_kv_heads']}KV` |
| **Vocab** | `{m['vocab_size']:,}` |
| **Context** | `{m['max_position_embeddings']:,}` |
| **Attention** | `{m['attn_impl'] or 'default'}` |
| **Disk** | `{m['size_mb']} MB` |
""")

    st.divider()

    # Generation parameters
    st.subheader("Parameters")
    max_new_tokens = st.slider("Max tokens", 16, 512, 128, step=16)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    top_k = st.slider("Top-k", 1, 200, 50, step=1)
    rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, step=0.05)
    suppress_eos = st.checkbox("Suppress EOS (force full length)")

    st.divider()

    # Unload
    if st.button("🗑 Clear model cache", use_container_width=True):
        load_model.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        st.success("Cache cleared")
        st.rerun()

# ── Main area ────────────────────────────────────────────────────

# Quick prompts — 2 rows, wraps on mobile
quick_prompts = [
    "বাংলাদেশের ইতিহাস",
    "আমার প্রিয় খাবার হলো",
    "ঢাকা শহরে",
    "বিজ্ঞান ও প্রযুক্তির",
    "শিক্ষার গুরুত্ব হলো",
]

qp_cols = st.columns(3)
for i, qp in enumerate(quick_prompts):
    if qp_cols[i % 3].button(qp, key=f"qp_{i}", use_container_width=True):
        st.session_state["prompt_text"] = qp

# Prompt input
prompt = st.text_area(
    "Prompt",
    value=st.session_state.get("prompt_text", ""),
    placeholder="এখানে বাংলায় লিখুন — type your Bengali prompt here...",
    height=100,
    label_visibility="collapsed",
)

# Generate buttons
col1, col2, col3 = st.columns([1, 1, 4])
generate_clicked = col1.button("⚡ Generate", type="primary",
                               use_container_width=True)
clear_clicked = col2.button("Clear results", use_container_width=True)

if clear_clicked:
    st.session_state.pop("results", None)
    st.rerun()

# Initialize results
if "results" not in st.session_state:
    st.session_state["results"] = []

# Generate
if generate_clicked and prompt.strip():
    with st.spinner("Loading model & generating..."):
        try:
            model, device, actual_params = load_model(selected_model["path"])
            result = generate_text(
                model=model,
                device=device,
                sp=sp,
                prompt=prompt.strip(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=rep_penalty,
                suppress_eos=suppress_eos,
            )
            result["model_name"] = selected_model["name"]
            result["actual_params_m"] = round(actual_params / 1e6, 1)
            result["prompt"] = prompt.strip()
            st.session_state["results"].insert(0, result)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
elif generate_clicked:
    st.warning("Enter a prompt first")

# ── Display results ──────────────────────────────────────────────

if not st.session_state["results"]:
    st.markdown(
        "<div style='text-align:center;color:#8b95a5;padding:60px;'>"
        "<div style='font-size:40px;margin-bottom:12px;'>✦</div>"
        "Select a model and enter a prompt to begin"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for r in st.session_state["results"]:
        param_label = ""
        if r.get("actual_params_m"):
            p = r["actual_params_m"]
            param_label = (f"{p/1000:.1f}B" if p >= 1000 else f"{p}M")

        with st.container(border=True):
            # Header with responsive layout
            param_display = ""
            if param_label:
                param_display = f" · {param_label}"
            st.markdown(
                f"<div class='result-card-header'>"
                f"<span class='model-label'>{r['model_name']}{param_display}</span>"
                f"<span class='stats-label'>"
                f"{r['new_tokens']} tok · {r['elapsed_s']}s · "
                f"{r['tokens_per_s']} tok/s</span>"
                f"</div>"
                f"<div class='result-card-body'>"
                f"<span class='prompt-echo'>{r['prompt']}</span> "
                f"<span class='generated-text'>{r['generated']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )