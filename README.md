# SSSL-HAR: Synthetic-Data-Driven Self-Supervised Learning for Flexible IMU Activity Recognition

[![IJCB 2025](https://img.shields.io/badge/Paper-IJCB%202025-blue.svg)](https://ieeexplore.ieee.org/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/) [![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A clean-room implementation and interactive laboratory for **SSSL-HAR** (Li et al., IJCB 2025). This framework eliminates the fundamental bottlenecks of data collection and rigid hardware placement in wearable Human Activity Recognition (HAR) by bridging **3D Virtual Kinematic Synthesis** and **Multi-View Contrastive Learning (MVCL)**.

---

## 💡 Why We Are Doing This (The Motivation & Impact)

### 1. The Real-World Healthcare & Fitness Sensing Bottleneck
Deploying wearable AI systems (for patient fall detection, rehabilitation monitoring, or elite athletic tracking) currently faces significant practical friction:
- **Data Scarcity & Laborious Annotation**: Gathering hundreds of hours of multi-sensor IMU recordings from diverse human subjects requires rigid protocol compliance and expensive manual labeling.
- **Hardware & Placement Sensitivity**: Deep learning algorithms trained on specific devices (e.g., 148Hz accelerometers attached to the left wrist) frequently experience massive accuracy drop-offs when an edge user deploys a different sensor (e.g., a 60Hz commercial smartwatch strapped slightly higher on the arm).

### 2. The Solution: Zero-Cost Synthetic Scaling & Universal Adaptation
By generating synthetic inertial readings directly from abundant 3D human kinematic surface trajectories (e.g., SMPL / AMASS motion capture datasets) and training via Multi-View Contrastive Learning (MVCL), **SSSL-HAR achieves zero-cost pre-training that generalizes across diverse devices**.
- **99% Reduction in Data Collection**: Requires only **10–30 minutes (~1–2 subjects)** of labeled target calibration data to fine-tune state-of-the-art recognition models.
- **Superior Generalization**: Surpasses models pre-trained on real-world unlabeled IMU datasets (e.g., scoring **88.15% accuracy** on PAMAP2 3ACC+3GYRO and **86.44%** on Custom Fitness monitoring).

---

## 🔬 What We Are Doing Here (Technical Architecture)

The system relies on two intuitive biomechanical observations:
1. **Spatial Sparsity $\rightarrow$ Virtual IMU Synthesis**: Trajectories of sparse anatomical points suffice for effective motion tracking. Using smoothed differential equations and local coordinate gravity transformations, we synthesize high-fidelity 3-axis accelerometer and gyroscope waveforms with injected physical non-inertial sensor noise (random walk bias + Gaussian noise).
2. **Temporal Coherence $\rightarrow$ Multi-View Contrastive Learning (MVCL)**: Distinct body sensors (wrist, chest, ankle) synchronously capture information about the exact same physical movement. By optimizing cross-view contrastive loss functions like **CroSSL (VICReg)** and **COCOA**, our multi-view 1D CNN encoders learn placement-invariant and activity-discriminating representations.

```
[3D Kinematic MoCap / SMPL] 
         │ 
         ├── (Smoothed Differentiation Eq. 1 & Gravity Rotation Eq. 2)
         ▼
[Virtual Multi-View IMU Signals] ──(Add PNP Sensor Bias & White Noise)──► [Synthetic Pretraining Set]
                                                                                  │
                                                                   (Stage 1: CroSSL / COCOA MVCL)
                                                                                  ▼
[Real Target Application] ───────(Minimal Labeled Finetuning Data)────────► [Deployable HAR Model]
```

---

## 🛠️ Repository & Module Overview

```
SSSL_HAR/
├── sssl_har/
│   ├── synthesis/         # Smoothed kinematic differentiation (Eq 1/2) & PNP physical noise modeling
│   ├── models/            # Multi-view 1D ConvNet encoders, 2-layer Aggregator & HAR classification head
│   ├── losses/            # MVCL contrastive losses (CroSSL/VICReg, COCOA) and baselines (SimCLR, CPC)
│   ├── methods/           # Unified training method adapters with random latent spatial masking
│   ├── data/              # Turnkey adapters and activity simulation engines for PAMAP2 & Custom Fitness
│   ├── engine/            # Two-stage pretraining and fine-tuning experimental orchestrator
│   └── utils/             # Benchmark metrics calculation and t-SNE representation visualizer
├── tests/                 # Automated validation test suite covering numerical precision and gradients
├── run_benchmark.py       # Command-line benchmark replication script (Tables 1 & 2, Figures 4-6)
└── app.py                 # Interactive Streamlit Web Laboratory & Bio-Sensing Dashboard
```

---

## 🚀 Quickstart Guide

### 1. Launch the Interactive Web Studio
Experience kinematic signal generation, live t-SNE clustering, and hardware robustness charts in your browser:
```powershell
streamlit run app.py
```

### 2. Run Full Quantitative Benchmark Suite
Execute the entire comparative evaluation matrix across all self-supervised learning paradigms via terminal:
```powershell
# Run validation test on PAMAP2 (3ACC + 3GYRO)
python run_benchmark.py --dataset pamap2 --sensor_config 3ACC+3GYRO

# Run evaluation on Custom Fitness dataset (25 classes)
python run_benchmark.py --dataset custom-25
```

### 3. Execute Automated Unit Test Suite
```powershell
python -m unittest discover tests
```
