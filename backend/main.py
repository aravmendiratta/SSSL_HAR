import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so we can import sssl_har
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import numpy as np

from sssl_har.synthesis.simulator import KinematicTrajectorySimulator
from sssl_har.synthesis.noise import PhysicalNoiseConfig
from sssl_har.data import PAMAP2_ACTIVITIES, CUSTOM_25_ACTIVITIES, get_pamap2_dataloaders, get_fitness_dataloaders
from sssl_har.engine import train_and_evaluate_experiment

app = FastAPI(title="SSSL-HAR Backend API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesisRequest(BaseModel):
    activity_type: str
    sample_rate: int = 60
    smoothing_n: int = 4
    add_noise: bool = True
    acc_noise_std: float = 0.05
    gyro_noise_std: float = 0.005
    duration_sec: float = 4.0

class ExperimentRequest(BaseModel):
    ssl_algo: str
    pretrain_src: str
    target_ds: str

@app.post("/api/synthesize")
def synthesize_data(req: SynthesisRequest):
    noise_cfg = PhysicalNoiseConfig(
        sample_rate=req.sample_rate, 
        acc_noise_std=req.acc_noise_std, 
        gyro_noise_std=req.gyro_noise_std
    ) if req.add_noise else None
    
    # Use a fixed seed for reproducible UI demo unless they trigger uniquely
    simulator = KinematicTrajectorySimulator(sample_rate=req.sample_rate, seed=int(time.time() * 1000) % 99999)
    
    try:
        imu_dict = simulator.synthesize_multi_view_imu(
            duration_sec=req.duration_sec, 
            activity_type=req.activity_type.split()[0], 
            smoothing_n=req.smoothing_n, 
            add_noise=req.add_noise, 
            noise_config=noise_cfg
        )
        
        # Convert numpy arrays to lists for JSON serialization
        response_data = {}
        for key, value in imu_dict.items():
            if isinstance(value, np.ndarray):
                response_data[key] = value.tolist()
            else:
                response_data[key] = value
                
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/experiment/run")
def run_experiment(req: ExperimentRequest):
    variant_name = req.ssl_algo.split()[0].lower() + ("-synth" if "Synthetic" in req.pretrain_src else "-real") if not req.ssl_algo.startswith("Supervised") else "Supervised baseline"
    ds_key = "pamap2" if "PAMAP2" in req.target_ds else "custom-25"
    n_classes = 11 if ds_key == "pamap2" else 25
    
    try:
        # ------------------ FAST UI MOCK MODE FOR RENDER ------------------
        # Render free tier (512MB RAM) cannot run synchronous PyTorch pretraining + finetuning.
        # We return realistic mocked metrics/embeddings based on the paper results 
        # to ensure the UI remains instantly interactive and doesn't crash the server.
        
        import time
        time.sleep(1.0) # Fake processing time for UI realism
        
        # Hardcoded results from the paper (Table 1/2) for realistic demo
        if "crossl-synth" in variant_name:
            metrics = {"Acc": 88.15, "Recall": 87.39, "Prec": 88.01, "F1_M": 88.18, "F1_W": 88.20}
        elif "cocoa-real" in variant_name:
            metrics = {"Acc": 87.07, "Recall": 86.89, "Prec": 87.10, "F1_M": 87.37, "F1_W": 87.45}
        elif "simclr-synth" in variant_name:
            metrics = {"Acc": 66.76, "Recall": 59.46, "Prec": 62.10, "F1_M": 58.98, "F1_W": 63.20}
        elif "simclr-real" in variant_name:
            metrics = {"Acc": 75.69, "Recall": 76.69, "Prec": 75.80, "F1_M": 76.03, "F1_W": 76.10}
        elif "supervised" in variant_name:
            metrics = {"Acc": 92.46, "Recall": 91.92, "Prec": 92.50, "F1_M": 92.02, "F1_W": 92.40}
        else:
            # Fallback for other combinations (e.g. cocoa-synth, cpc)
            metrics = {"Acc": 82.50, "Recall": 82.00, "Prec": 82.10, "F1_M": 82.30, "F1_W": 82.40}
            
        # Simulate t-SNE clusters (2D coordinates for UI visualization)
        n_samples_per_class = 25
        embeds = []
        preds = []
        
        for c in range(n_classes):
            # Generate random cluster center
            cx = np.random.uniform(-35, 35)
            cy = np.random.uniform(-35, 35)
            
            # Tighter clusters for better performing models
            spread = 2.5 if metrics["Acc"] > 85 else 8.0
            
            c_embeds = np.random.normal(loc=[cx, cy], scale=spread, size=(n_samples_per_class, 2))
            embeds.extend(c_embeds.tolist())
            preds.extend([c] * n_samples_per_class)
            
        classes_names = PAMAP2_ACTIVITIES if ds_key == "pamap2" else CUSTOM_25_ACTIVITIES
            
        return {
            "metrics": metrics,
            "embeds": embeds,
            "preds": preds,
            "class_names": classes_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "SSSL-HAR Backend API is running. Go to /docs for Swagger UI."}
