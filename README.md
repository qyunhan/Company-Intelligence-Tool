# Decision-Conditioned Company Intelligence

A multi-lens framework for structure discovery, peer analysis, and explainable risk — built on large-scale firm-level data from the CHAMPIONS Group dataset.

**Course:** NUS · Cat A: Data Warriors  
**Team:** Lim Shi Ying, Qian Yunhan, Yang Xiuli

---

## Overview

Rather than treating companies as standalone entities, this system groups firms by shared operational, organisational, and firmographic characteristics — and crucially, recognises that *who your peers are depends on the question you're asking*. A firm may be structurally similar to one set of companies, digitally comparable to another, and governance-aligned with yet another.

The framework produces three parallel peer views, each grounded in unsupervised learning and translated into interpretable, auditable risk signals.

---

## The Three Analytical Lenses

| Lens | Business Question | Method | Peer Type |
|------|------------------|--------|-----------|
| Firmographic | "Who looks like me structurally?" | KMeans (k=3) | Archetype peers |
| Technology Intensity | "Who has similar digital maturity?" | HDBSCAN | Transformation peers |
| Corporate Complexity | "Who has an unusual ownership structure?" | KMeans | Governance peers |

Each lens has its own feature matrix, clustering approach, and validation strategy — chosen to match the geometry of that feature space rather than applied uniformly.

---

## Key Results

- Firmographic and complexity lenses produced stable, interpretable clusters validated by silhouette analysis and Adjusted Rand Index (ARI) across random seeds
- Technology lens used HDBSCAN to avoid forced assignment; firms in sparse regions are conservatively labelled as noise
- Corporate complexity lens achieved the highest silhouette scores (~0.41), reflecting the inherently discrete nature of ownership structures
- Cross-lens contradiction detection flagged firms with strong quartile divergence across dimensions (e.g. high structural scale but low tech intensity), surfacing latent strategic tensions
- Four named risk archetypes generated for each firm: Scale Imbalance Risk, IT Underinvestment Risk, Complexity Mismatch Risk, Data Opacity Risk
- LLM explanation layer (via LangChain + HuggingFace) synthesises structured outputs into human-readable narratives for non-technical stakeholders

---

## Skills & Techniques

**Unsupervised Learning**
`KMeans` `HDBSCAN` `Silhouette Analysis` `Adjusted Rand Index` `Clustering Validation` `Density-Based Clustering`

**Dimensionality Reduction & Visualisation**
`PCA` `UMAP` `Variance Decomposition` `Low-Dimensional Embeddings`

**Feature Engineering**
`Log Transformation` `Winsorization` `Z-Score Standardisation` `One-Hot Encoding` `Top-k Encoding` `Composite Score Construction` `Percentile Ranking` `Ownership Opacity Scoring`

**Explainability & Risk**
`Rule-Based Risk Archetypes` `Cross-Lens Contradiction Detection` `Peer-Relative Percentile Benchmarking` `Quartile Ranking` `Explainable AI (XAI)`

**NLP & LLM Integration**
`LangChain` `HuggingFace` `LLM Narrative Synthesis` `Structured Prompt Design`

**Data & Tools**
`Python` `Pandas` `NumPy` `Scikit-learn` `Scipy` `Seaborn` `Matplotlib`

---

## Dataset

> ⚠️ **Dataset not included** — the CHAMPIONS Group dataset is proprietary and cannot be redistributed.

- 8,559 company records across 74 variables
- Features include: revenue, employee counts, market value, IT expenditure, device counts, SIC industry codes, legal status, ownership structure, and corporate family relationships
- Extensive preprocessing applied: zero-as-missing treatment, log transforms, range-to-midpoint conversion, restricted-data flagging

---

## Repository Structure

```
├── CAT_A_Data_Warriors.ipynb   # Full analysis pipeline
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.10 required.**

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas==1.5.0`, `scikit-learn==1.2.0`, `umap==0.5.11`, `langchain==0.2.6`, `langchain-huggingface==0.0.3`

---

## Key Findings

- **Similarity is lens-dependent** — firmographic peers and technology peers for the same firm rarely overlap, validating the multi-lens design over single-view segmentation
- **Algorithm selection should match feature geometry** — forcing KMeans on the technology lens would have over-fragmented a continuous, gradient-based space; HDBSCAN was more appropriate
- **Cross-lens contradictions are informative signals** — mismatches (e.g. high complexity, low scale) surface governance and coordination risks invisible within any single lens
- **Separation of structure discovery from interpretation** — unsupervised models identify peer structure only; all risk signals are derived via transparent, rule-based logic downstream, ensuring auditability
- **Dual percentile framing** — both overall and within-cluster percentiles are reported per firm, enabling both global benchmarking and local peer comparison simultaneously
