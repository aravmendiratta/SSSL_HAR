"""
Evaluation metric calculations for Human Activity Recognition tasks.
Implements Accuracy, Macro Recall, Macro Precision, Macro F1 (F1_M), and Weighted F1 (F1_W) as in Tables 1 & 2.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_har_metrics(y_true: Union[np.ndarray, List[int]], y_pred: Union[np.ndarray, List[int]]) -> Dict[str, float]:
    """
    Computes standard HAR benchmark evaluation percentages:
    - Acc: Overall classification accuracy
    - Recall: Macro-averaged recall across all activity classes
    - Prec: Macro-averaged precision across all activity classes
    - F1_M: Macro-averaged F1 score
    - F1_W: Weighted-averaged F1 score accounting for class imbalances
    
    Returns:
        Dictionary of percentage scores (0.0 to 100.0) rounded to 2 decimal places.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    acc = accuracy_score(y_true_arr, y_pred_arr) * 100.0
    
    # Macro metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    prec_macro *= 100.0
    rec_macro *= 100.0
    f1_macro *= 100.0
    
    # Weighted F1
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    f1_weighted *= 100.0
    
    return {
        "Acc": round(acc, 2),
        "Recall": round(rec_macro, 2),
        "Prec": round(prec_macro, 2),
        "F1_M": round(f1_macro, 2),
        "F1_W": round(f1_weighted, 2),
    }


def format_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Transforms a dictionary mapping method names (e.g., 'CroSSL-synth') to metric dictionaries
    into a polished Pandas DataFrame matching Table 1 and Table 2 format.
    """
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    desired_order = ["Acc", "Recall", "Prec", "F1_M", "F1_W"]
    existing_cols = [col for col in desired_order if col in df.columns]
    return df[existing_cols]
