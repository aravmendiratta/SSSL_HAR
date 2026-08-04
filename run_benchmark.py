"""
CLI Benchmark Runner for SSSL-HAR.
Executes experiments across baseline SSL variants (SimCLR, CPC, COCOA, CroSSL) on real vs synthetic data,
generating comparative evaluation tables matching Table 1 (PAMAP2) and Table 2 (Custom Fitness).
"""

import argparse
import time
from typing import Dict
from sssl_har.data import get_pamap2_dataloaders, get_fitness_dataloaders
from sssl_har.engine import train_and_evaluate_experiment
from sssl_har.utils import format_metrics_table


def run_benchmark_suite(
    dataset_type: str = "pamap2",
    sensor_config: str = "3ACC+3GYRO",
    num_classes: int = 11,
    pretrain_epochs: int = 10,
    finetune_epochs: int = 15,
    fast_dev_run: bool = False
):
    print(f"\n==========================================================================")
    print(f"      SSSL-HAR Benchmark Study | Dataset: {dataset_type.upper()} ({sensor_config})")
    print(f"==========================================================================")
    
    if fast_dev_run:
        print("[Notice] Fast dev run active: scaling down epochs for instantaneous verification.")
        pretrain_epochs = 2
        finetune_epochs = 3
        
    methods_list = [
        "SimCLR-real", "SimCLR-synth",
        "CPC-real", "CPC-synth",
        "COCOA-real", "COCOA-synth",
        "CroSSL-real", "CroSSL-synth",
        "Supervised baseline"
    ]
    
    # Initialize dataloaders
    start_t = time.time()
    if "pamap2" in dataset_type.lower():
        pretrain_loader, finetune_loader, test_loader = get_pamap2_dataloaders(
            batch_size=32, sensor_config=sensor_config, num_train_samples=200 if fast_dev_run else 400
        )
        num_classes = 11
    else:
        num_classes = int(dataset_type.split("-")[-1]) if "-" in dataset_type else 25
        pretrain_loader, finetune_loader, test_loader = get_fitness_dataloaders(
            batch_size=32, num_classes=num_classes, num_train_samples=200 if fast_dev_run else 400
        )
        
    results_matrix = {}
    for variant in methods_list:
        print(f" -> Evaluating [{variant:20s}] ... ", end="", flush=True)
        t0 = time.time()
        metrics, _, _ = train_and_evaluate_experiment(
            method_variant=variant,
            dataset_name=dataset_type,
            sensor_config=sensor_config,
            num_classes=num_classes,
            pretrain_loader=pretrain_loader,
            finetune_loader=finetune_loader,
            test_loader=test_loader,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            verbose=False
        )
        t1 = time.time()
        print(f"Done in {t1 - t0:.1f}s | Acc: {metrics['Acc']:6.2f}% | F1_M: {metrics['F1_M']:6.2f}%")
        results_matrix[variant] = metrics
        
    df = format_metrics_table(results_matrix)
    print(f"\n--- Final Performance Summary ({dataset_type.upper()} - {sensor_config}) ---")
    print(df.to_string())
    print(f"==========================================================================\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSSL-HAR IJCB 2025 Paper Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="pamap2", choices=["pamap2", "custom-25", "custom-43"], help="Target benchmark dataset")
    parser.add_argument("--sensor_config", type=str, default="3ACC+3GYRO", choices=["3ACC+3GYRO", "3ACC"], help="Sensor view configuration")
    parser.add_argument("--pretrain_epochs", type=int, default=12, help="Number of SSL pre-training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=15, help="Number of supervised fine-tuning epochs")
    parser.add_argument("--fast_dev_run", action="store_true", help="Perform rapid validation test")
    args = parser.parse_args()
    
    run_benchmark_suite(
        dataset_type=args.dataset,
        sensor_config=args.sensor_config,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        fast_dev_run=args.fast_dev_run
    )
