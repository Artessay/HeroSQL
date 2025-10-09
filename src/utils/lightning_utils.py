import os
import json
from pytorch_lightning.utilities import rank_zero_only

from src.metrics import evaluate_and_save

@rank_zero_only
def print_only_on_rank_zero(message):
    print(message)

@rank_zero_only
def evaluate_and_save_on_rank_zero(score_list, label_list, save_path: str):
    evaluate_and_save(score_list, label_list, save_path)

# @rank_zero_only
# def save_evaluation_result(test_metrics_list, save_path: str):
#     if isinstance(test_metrics_list, list) and len(test_metrics_list) > 0:
#         metrics = test_metrics_list[0]
#     elif isinstance(test_metrics_list, dict):
#         metrics = test_metrics_list
#     else:
#         metrics = {}

#     output_metrics = {
#         "Accuracy": metrics.get('test_accuracy'),
#         "Precision": metrics.get('test_precision'),
#         "Recall": metrics.get('test_recall'),
#         "F1-Score": metrics.get('test_f1'),
#         "AUROC": metrics.get('test_auc_roc'),
#         "AUPRC": metrics.get('test_auc_pr'),
#     }

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, 'w') as f:
#         json.dump(output_metrics, f, indent=4)