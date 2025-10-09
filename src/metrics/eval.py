import os
import json
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def evaluate_model(score_list, label_list): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate metrics
    metrics = evaluate_binary_model(score_list, label_list)
    return metrics


def evaluate_and_save(score_list, label_list, save_path):
    metrics = evaluate_model(score_list, label_list)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_binary_model(score_list, label_list): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)

    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    
    # Validate inputs
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # ===== Basic Metrics Calculation =====
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    brier_score = brier_score_loss(y_true, y_scores)
    
    # ===== Best F1 Score Calculation =====
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8) # Avoid division by zero
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    # thresholds length = len(precisions)-1, 所以用条件判断一下
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    
    # Construct results dictionary
    metrics = {
        'AUPRC': auprc,
        'AUROC': auroc,
        'F1 Score': best_f1,
        'Brier Score': brier_score,
        'Threshold': best_threshold,
    }
    
    # Print evaluation results
    print("Binary Classification Model Evaluation Results:")
    print(f"  PR AUC: {auprc:.4f}")
    print(f"  ROC AUC: {auroc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    
    return metrics


def evaluate_model_with_threshold(score_list, label_list, threshold=0.5): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate metrics at high recall threshold
    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_model_with_high_recall(score_list, label_list, recall_threshold=0.9): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """

    # Validate inputs
    if len(score_list) != len(label_list):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # inverse score_list and label_list
    score_list = [1 - score for score in score_list]
    label_list = [1 - label for label in label_list]
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(label_list, score_list)
    
    # Find the threshold at high recall
    high_recall_index = np.where(recall_curve >= recall_threshold)[0][-1]
    threshold = threshold_curve[high_recall_index]

    # Calculate metrics at high recall threshold
    metrics = evaluate_binary_model_with_threshold(score_list, label_list, threshold)
    return metrics


def evaluate_binary_model_with_threshold(score_list, label_list, threshold): 
    """
    Evaluate the performance of a binary classification model by calculating various metrics
    and optionally plotting ROC and Precision-Recall curves.
    
    Parameters:
    score_list (list): List of predicted probabilities (scores between 0 and 1)
    label_list (list): List of true binary labels (0 or 1)
    threshold (float): Threshold for converting probabilities to binary predictions, default is 0.5
    plot_curves (bool): Whether to plot ROC and Precision-Recall curves
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays
    y_scores = np.array(score_list)
    y_true = np.array(label_list)
    
    # Validate inputs
    if len(y_scores) != len(y_true):
        raise ValueError("Length of predicted scores and true labels must be the same")
    
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate basic evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    
    # Construct results dictionary
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'PR AUC': pr_auc,
        'ROC AUC': roc_auc,
        'Confusion Matrix': {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp
        },
    }
    
    # Print evaluation results
    print("Binary Classification Model Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return metrics

if __name__ == '__main__':
    score_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    label_list = [0, 0, 1, 0, 0, 1, 1, 0, 1]
    metrics = evaluate_model_with_high_recall(score_list, label_list)
    print(metrics)