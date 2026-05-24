# 🐍 MAMBA From Scratch — Deep Learning Project

> **ENIS 2025–2026 · Deep Learning Course**  
> Implementation of MAMBA (Selective State Space Model) from scratch in PyTorch,  
> with a full comparison against a Transformer baseline on character-level language modeling.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [AI Tools Used](#ai-tools-used)
- [References](#references)

---

## 🔍 Overview

This project implements **MAMBA (S6 — Selective State Space Model)** completely from scratch in PyTorch, without using the official `mamba-ssm` package.

**Key contributions:**
- ✅ Full MAMBA block: `SelectiveSSM` + `Conv1D` + gating
- ✅ **Parallel Associative Scan** — O(log L) depth instead of O(L) sequential loop
- ✅ Transformer baseline (mini GPT) for fair comparison
- ✅ Character-level language modeling on TinyShakespeare
- ✅ Inference speed benchmark (tokens/second)
- ✅ Complexity analysis: MAMBA vs Transformer vs LSTM

**Why MAMBA?**

| Model | Time Complexity | Memory | Parallelizable |
|-------|----------------|--------|----------------|
| LSTM | O(L) | O(1) | ❌ |
| Transformer | O(L²) | O(L²) | ✅ |
| **MAMBA (S6)** | **O(L log L)** | **O(L)** | **✅** |

For L=4096: Transformer needs **16.7M ops** · MAMBA needs only **49K ops** → **341× faster**

---

## 📊 Results

| Metric | MAMBA | Transformer |
|--------|-------|-------------|
| **Best Val Loss** | **1.686** | 2.010 |
| **Best Perplexity** | **5.4** | 7.5 |
| Parameters | 475,520 | 818,048 |
| Avg Step Time | 134.8 ms | 16.7 ms |
| Total Train Time | ~754s | ~97s |

> MAMBA achieves **28% better perplexity** with **42% fewer parameters** than the Transformer.

### Training Curves
![Training Results](full_results.png)

### Complexity Scaling
![Complexity Analysis](complexity_analysis.png)

### Generated Text Sample (MAMBA)
```
HAMLET:
the continued the ground be such a one that be so.
POLIXENES: Then the father, I have to what how my father
That would away the may life, here stay to such in the sorrow...
```

---

## 📁 Repository Structure

```
mambaDeep_Learning/
├── mamba_improved.ipynb        # Main notebook — full MAMBA implementation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── full_results.png            # Training curves & comparison charts
├── complexity_analysis.png     # O(L log L) vs O(L²) scaling plots
└── mamba_v2_shakespeare.pt     # Pre-trained model checkpoint
```

---

## ⚡ Quick Start

### Option 1 — Google Colab (recommended)
1. Open `mamba_improved.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: `Ctrl+F9`

### Option 2 — Local setup
```bash
# Clone the repository
git clone https://github.com/myriamBenAbd0607/mambaDeep_Learning.git
cd mambaDeep_Learning

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook mamba_improved.ipynb
```

### Requirements
```
torch>=2.0.0
numpy
matplotlib
```

---

## 🏗️ Architecture

### MAMBA Block
```
Input x (B, L, d_model)
       │
   LayerNorm
       │
   in_proj ──────────────┐
       │                 │
    x_main             z (gate)
       │
   Conv1D (local context, kernel=4)
       │
    SiLU
       │
  SelectiveSSM (S6)
   ├── A: fixed log-parameterized  (d_inner, d_state)
   ├── Δ, B, C: input-dependent   ← KEY INNOVATION
   └── Parallel Associative Scan  ← O(log L) depth
       │
    × SiLU(z)   ← gating
       │
   out_proj
       │
  + residual
       │
Output y (B, L, d_model)
```

### Parallel Scan — The Core Idea
```
Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t

Naive loop:  O(L) sequential steps — no GPU parallelism
Parallel:    O(log L) depth via divide-and-conquer

L=8 example:
  Level 1: combine pairs     → 4 parallel ops
  Level 2: combine pairs     → 2 parallel ops
  Level 3: combine pairs     → 1 op
  Fill even positions        → 4 parallel ops
  Total: 3 passes instead of 8!
```

---

## 🤖 AI Tools Used

| Tool | Usage |
|------|-------|
| **Claude (Anthropic)** | Code generation, architecture design, explanations |
| **GitHub Copilot** | Code completion |
| **ChatGPT** | Cross-checking math derivations |
| **Gamma.app** | AI-powered presentation generation |
| **DALL-E** | AI-generated images for slides |

---

## 📚 References

1. **Gu & Dao (2023)** — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*  
   [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

2. **Gu et al. (2021)** — *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*  
   [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

3. **Official MAMBA repository**  
   [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

4. **Karpathy (2022)** — *nanoGPT / char-rnn* (TinyShakespeare dataset)  
   [github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)

---

## 👥 Authors

- **Student** — ENIS, Deep Learning Course 2025–2026
- **Colleague** — ENIS, Deep Learning Course 2025–2026

---

*This project was built as part of the Deep Learning course at ENIS (École Nationale d'Ingénieurs de Sfax).*