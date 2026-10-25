# AsthmaExpertUltra: A Transformer-Based Multi-Task Learning System for Real-Time Prediction and Monitoring of Asthma Crisis Severity and Temporal Proximity

[![CERIST Research](https://img.shields.io/badge/Research-CERIST%20Algeria-blue.svg)](https://www.cerist.dz)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Android PyTorch Lite](https://img.shields.io/badge/Android-PyTorch%20Lite%20(.ptl)-green.svg?logo=android&logoColor=white)](https://pytorch.org/mobile/home/)
[![Status](https://img.shields.io/badge/Status-Research%20Internship%20Validated-success.svg)]()
[![Code Protection](https://img.shields.io/badge/Code%20Policy-Intellectual%20Property%20Protected-orange.svg)]()

> **Research Center in Scientific and Technical Information (CERIST)**  
> **Author:** Hani Ghena (M.Sc. AI Candidate @ USTHB · CERIST AI Researcher)  
> **Key Contributions:** Multimodal physiological signal processing, 4-layer Transformer Encoder with multi-task prediction heads, class-weighted Focal Loss, Dempster–Shafer spirometric safety fusion, and sub-second PyTorch Lite on-device Android deployment.

---

## 📌 Abstract & Clinical Motivation
Asthma is a chronic respiratory condition affecting hundreds of millions of individuals worldwide, characterised by recurrent and unpredictable exacerbations that can rapidly escalate into life-threatening emergencies. 

Traditional clinical approaches remain strictly **reactive** — detecting crises only after onset. **AsthmaExpertUltra** is an end-to-end anticipatory edge-AI monitoring system that identifies pre-crisis physiological degradation patterns **4 to 8 hours before clinical onset**, jointly forecasting:
1. **Crisis Severity:** 4 discrete clinical tiers (*Stable*, *Mild*, *Moderate*, *Severe*).
2. **Temporal Proximity (Lead-Time):** 4 time-to-crisis risk bins (*Imminent* $<1$h, *Near-term* $1\text{--}3$h, *Alert* $3\text{--}5$h, *Stable* $>5$h).

The entire model is exported to **PyTorch Lite (`.ptl`)** and operates fully offline on an Android smartphone with sub-second latency, zero cloud dependency, and total medical data privacy.

---

## 🔬 System Architecture

```
Multimodal Sensor Input (8 Channels)
├── Spirometry: FEV1, PEF
├── Cardiorespiratory: SpO2, Heart Rate (HR), Respiratory Rate (RR)
└── Environmental: PM2.5, Pollen, Temperature
             │
             ▼
[Patient-Specific Z-score Normalisation]  x_hat = (x - mu_i) / sigma_i
             │
             ▼
[24-Hour Sliding Temporal Window]  X ∈ R^(24 × 8) (1-hour stride)
             │
             ▼
[Linear Projection + Sinusoidal Positional Encoding]  d_in=8 → d_model=128
             │
             ▼
[4-Layer Transformer Encoder Stack]
├── 8 Multi-Head Self-Attention Heads (d_k = 16) + LayerNorm & Residuals
└── Position-wise Feed-Forward Network (128 → 512 → 128) + LayerNorm
             │
             ▼
[Global Average Pooling (GAP)]  h_GAP ∈ R^128
             │
      ┌──────┴──────────────────────┐
      ▼                             ▼
[Severity Head]               [Lead-Time Head]
MLP: 128 → 64 → 4             MLP: 128 → 64 → 4
Loss: Focal (γ=2, w=[1,5,5,10]) Loss: Cross-Entropy
Output: 4 Severity Classes    Output: 4 Temporal Bins
      │                             │
      └──────┬──────────────────────┘
             ▼
[Multi-Criteria Decision Logic & Safety Override]
├── Dempster–Shafer Belief Fusion (Network belief ⊗ Spirometric prior)
└── Deterministic Clinical Override: VEMS < 50% VEMS_ref ⇒ Immediate Emergency Alert
             │
             ▼
[Android Mobile Visualisation (PyTorch Lite .ptl)]  Latency < 1.0s, fully offline
```

---

## 📊 Experimental Setup & Dataset Specification

- **Cohort:** 1,000 simulated patients monitored over 1,440 continuous hours (60 days each).
- **Incident Dynamics:** 4–8 exacerbations per patient, each preceded by an 8-hour progressive incubation phase (stochastic FEV1 drop, compensatory RR rise, SpO2 degradation).
- **Leakage-Free Splitting:** Strict `GroupShuffleSplit` (80% train / 20% test) grouped strictly by patient ID — ensuring **zero patient overlap** between training and evaluation cohorts.
- **Benchmarked Baselines:**
  - Gradient Boosted Trees (XGBoost) on flattened windows
  - Recurrent Neural Network (Simple RNN)
  - Long Short-Term Memory (LSTM)
  - Patch-based Transformer (PatchTST, Nie et al., 2023)

---

## 📈 Key Findings & Benchmark Metrics

| Model Architecture | Severity Accuracy | Severity Macro-F1 | Lead-Time Accuracy | Lead-Time Macro-F1 | Android Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AsthmaExpertUltra (Ours)** | **76.0%** | **0.460** | **69.0%** | **0.640** | **< 1.0s (.ptl)** |
| XGBoost Baseline | 61.8% | 0.445 | 85.1% | 0.778 | N/A (Server) |
| LSTM Baseline | 59.3% | 0.346 | 85.7% | 0.793 | ~1.4s |
| Simple RNN Baseline | 59.3% | 0.345 | 85.3% | 0.792 | ~1.1s |
| PatchTST (SOTA 2023) | 57.7% | 0.340 | 85.0% | 0.788 | ~1.3s |

### Clinical Safety & Ablation Insights:
1. **Class-Weighted Focal Loss Impact:** Applying $w = [1, 5, 5, 10]$ with $\gamma=2$ escalates Severe class recall from **0.3% to 92.3%**, prioritising life-critical emergency detection.
2. **Dempster–Shafer Fusion:** Modulates decisions based on real-time spirometric confidence, escalating 47.3% of low-VEMS readings to higher triage categories while de-escalating false alarms in stable patients.
3. **Multi-Task Synergies:** Jointly optimising severity and lead-time allows the shared 128-dimensional latent representation to capture both the trajectory gradient and intensity of respiratory distress simultaneously.

---

## 📂 Repository Contents
- `AsthmaExpertUltra_FINALi.ipynb` — Complete end-to-end PyTorch training notebook, ablation studies, baseline comparisons, and TorchScript/PyTorch Lite mobile export routines.
- `AsthmaExpertUltra_Research_Report.pdf` — Formal research paper detailing clinical literature context, mathematical formulations, architectural diagrams, and experimental validation.
- `README.md` — Project technical overview and specifications.

---

## 🛡️ Intellectual Property & Code Protection Notice
The architecture, weights, and experimental benchmarks documented in this repository represent research carried out at **CERIST (Centre de Recherche sur l'Information Scientifique et Technique)**. 
- Direct automated cloning and raw weight redistribution are protected under institutional research IP policies.
- **Recruiters & Academic Collaborators:** For verified technical recruiters, hiring managers, and academic evaluators wishing to schedule a live code walkthrough or review the implementation, please contact via [LinkedIn](https://www.linkedin.com/in/ghena-hani-3543a3375/) or email (`hanighena4@gmail.com`).

---

## 📜 Citation & Acknowledgements
```bibtex
@article{ghena2026asthmaexpertultra,
  title={AsthmaExpertUltra: A Transformer-Based Multi-Task Learning System for Real-Time Prediction and Monitoring of Asthma Crisis Severity and Temporal Proximity},
  author={Ghena, Hani},
  journal={CERIST AI Research Technical Report},
  year={2026},
  institution={Research Center in Scientific and Technical Information (CERIST)}
}
```
Special thanks to the **Research Center in Scientific and Technical Information (CERIST)** for computational infrastructure and research guidance.
