# 🐍 MAMBA From Scratch — PyTorch Implementation

> Deep Learning Project · ENIS 2025–2026

Implementation complète de l'architecture **MAMBA (Selective State Space Model)** en PyTorch pur, avec comparaison contre un Transformer baseline sur la tâche de modélisation de langage au niveau caractère (TinyShakespeare).

---

## 🎯 Objectif

Comprendre et implémenter MAMBA from scratch en partant des fondations mathématiques (SSM continu → discret → sélectif), jusqu'à l'entraînement et la comparaison avec un Transformer.

---

## 🏗️ Architecture

```
Input tokens
     │
Embedding (vocab → d_model)
     │
  ×N_LAYERS
┌─────────────────────────┐
│       MambaBlock        │
│  LayerNorm              │
│  in_proj (×2)           │
│  Conv1D (local context) │
│  SiLU                   │
│  SelectiveSSM (S6)      │
│    ├── A (fixed, log)   │
│    ├── B, C, Δ (input)  │
│    └── Parallel Scan    │
│  × SiLU(z) gate         │
│  out_proj + residual    │
└─────────────────────────┘
     │
LayerNorm → LM Head → Logits
```

### Ce qui rend MAMBA "Sélectif" (S6)

Les SSM classiques (S4) utilisent des matrices A, B, C **fixes**. L'innovation MAMBA : **A, B, C et Δ sont toutes fonctions de l'entrée** `u_t`, ce qui permet au modèle de sélectivement mémoriser ou oublier selon le contenu.

---

## ⚡ Parallel Associative Scan

L'algorithme clé du projet : remplacer la récurrence séquentielle O(L) par un scan parallèle O(log L).

```
L=8:  h0  h1  h2  h3  h4  h5  h6  h7
Level1: h01  h23  h45  h67        (4 ops parallèles)
Level2:  h0123    h4567            (2 ops parallèles)
Level3:   h01234567                (1 op)
→ Seulement 3 passes GPU pour L=8, log₂(L) en général
```

---

## 📊 Complexité

| Modèle | Temps | Mémoire | Parallélisable |
|--------|-------|---------|----------------|
| RNN/LSTM | O(L) | O(1) | ❌ Séquentiel |
| Transformer | O(L²) | O(L²) | ✅ Totalement parallèle |
| S4 (SSM classique) | O(L log L) | O(L) | ✅ |
| **MAMBA (S6)** | **O(L log L)** | **O(L)** | ✅ **Parallel scan** |

Pour L=4096 : Transformer → 16,7M opérations · MAMBA → 49K opérations (**×341 plus rapide**)

---

## 🗂️ Contenu du notebook

| Section | Description |
|---------|-------------|
| 1. Setup | Imports, device, seeds |
| 2. Dataset | TinyShakespeare, tokenisation caractère, split train/val |
| 3. SSM Theory | Fondations mathématiques, ZOH discretization |
| 4. Parallel Scan | Implémentation + vérification vs scan séquentiel |
| 5. Architecture | `SelectiveSSM`, `MambaBlock`, `MambaLM` |
| 6. Transformer | Baseline GPT-style pour comparaison |
| 7. Training | AdamW + CosineAnnealing, 5000 itérations |
| 8. Résultats | Loss, perplexity, vitesse, comparaisons visuelles |
| 9. Génération | Démo de génération de texte Shakespeare |
| 10. Complexité | Analyse et visualisation des courbes de scaling |
| 11. Summary | Conclusions et sauvegarde du checkpoint |

---

## 🚀 Installation & Usage

### Prérequis

```bash
pip install -r requirements.txt
```

```
# requirements.txt
torch>=2.0.0
numpy
matplotlib
```

### Lancer le notebook

```bash
jupyter notebook mamba_final_complete.ipynb
```

Ou ouvrir directement dans **VS Code** avec l'extension Jupyter.

> ⚠️ L'entraînement est significativement plus rapide avec un GPU. Sans GPU, s'attendre à plusieurs heures pour 5000 itérations.

---

## 📈 Résultats

Les deux modèles sont entraînés dans les mêmes conditions (même `d_model=128`, 4 layers, 5000 itérations, AdamW lr=3e-4) :

- MAMBA atteint une **perplexité comparable** au Transformer avec moins de paramètres
- MAMBA présente un **meilleur scaling** sur les longues séquences (O(L log L) vs O(L²))
- Le **parallel scan** est la clé d'ingénierie : profondeur O(log L) sur GPU

---

## 🛠️ Outils utilisés

| Outil | Usage |
|-------|-------|
| **PyTorch** | Implémentation du modèle |
| **Google Colab** | Entraînement avec GPU |
| **VS Code** | Développement local |
| **Claude (Anthropic)** | Génération de code, explications, design d'architecture |
| **GitHub Copilot** | Complétion de code |
| **ChatGPT** | Vérification des dérivations mathématiques |

---

## 📚 Références

- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Gu et al. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Karpathy — [char-rnn / TinyShakespeare](https://github.com/karpathy/char-rnn)
- Dépôt officiel MAMBA : [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
