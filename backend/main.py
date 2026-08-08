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
        if ds_key == "pamap2":
            pre_L, fine_L, test_L = get_pamap2_dataloaders(batch_size=32, num_train_samples=250)
            classes_names = PAMAP2_ACTIVITIES
        else:
            pre_L, fine_L, test_L = get_fitness_dataloaders(batch_size=32, num_classes=25, num_train_samples=250)
            classes_names = CUSTOM_25_ACTIVITIES
            
        metrics, embeds, preds = train_and_evaluate_experiment(
            method_variant=variant_name, dataset_name=ds_key, num_classes=n_classes,
            pretrain_loader=pre_L, finetune_loader=fine_L, test_loader=test_L,
            pretrain_epochs=5, finetune_epochs=8
        )
        
        return {
            "metrics": metrics,
            "embeds": embeds.tolist() if isinstance(embeds, np.ndarray) else embeds,
            "preds": preds.tolist() if isinstance(preds, np.ndarray) else preds,
            "class_names": classes_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
