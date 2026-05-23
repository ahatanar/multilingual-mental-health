"""Evaluation metrics for depression classification."""

from typing import List, Dict


class EvaluationMetrics:
    """
    Computes confusion matrix and derived metrics from ground truth vs predictions.
    
    Labels: "depressed" (positive) vs "not depressed" (negative).
    """

    @staticmethod
    def compute(results: List[Dict]) -> Dict:
        """
        Compute all metrics from a list of result dicts.
        
        Each result dict must have:
            - ground_truth: "depressed" or "not depressed"
            - prediction: "depressed", "not depressed", "unclear", or "error"
            
        Returns:
            Dict with confusion matrix, precision, recall, F1, accuracy, 
            and per-entry breakdown.
        """
        tp = fp = tn = fn = 0
        unclear_count = 0
        error_count = 0

        for r in results:
            gt = r["ground_truth"]
            pred = r["prediction"]

            if pred in ("unclear", "error"):
                if pred == "unclear":
                    unclear_count += 1
                else:
                    error_count += 1
                continue

            if gt == "depressed" and pred == "depressed":
                tp += 1
            elif gt == "not depressed" and pred == "depressed":
                fp += 1
            elif gt == "not depressed" and pred == "not depressed":
                tn += 1
            elif gt == "depressed" and pred == "not depressed":
                fn += 1

        total_classified = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_classified if total_classified > 0 else 0.0

        return {
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "total_samples": len(results),
            "total_classified": total_classified,
            "unclear_responses": unclear_count,
            "error_responses": error_count,
        }

    @staticmethod
    def format_report(metrics: Dict, model_name: str) -> str:
        """Format metrics into a human-readable report string."""
        cm = metrics["confusion_matrix"]
        lines = [
            f"{'='*50}",
            f"  EVALUATION REPORT: {model_name}",
            f"{'='*50}",
            f"",
            f"  Confusion Matrix:",
            f"  ┌─────────────────┬──────────┬──────────────┐",
            f"  │                 │ Pred DEP │ Pred NOT DEP │",
            f"  ├─────────────────┼──────────┼──────────────┤",
            f"  │ Actual DEP      │ TP: {cm['true_positives']:4d} │ FN: {cm['false_negatives']:8d} │",
            f"  │ Actual NOT DEP  │ FP: {cm['false_positives']:4d} │ TN: {cm['true_negatives']:8d} │",
            f"  └─────────────────┴──────────┴──────────────┘",
            f"",
            f"  Precision:  {metrics['precision']:.4f}",
            f"  Recall:     {metrics['recall']:.4f}",
            f"  F1 Score:   {metrics['f1_score']:.4f}",
            f"  Accuracy:   {metrics['accuracy']:.4f}",
            f"",
            f"  Total samples:      {metrics['total_samples']}",
            f"  Classified:         {metrics['total_classified']}",
            f"  Unclear responses:  {metrics['unclear_responses']}",
            f"  Error responses:    {metrics['error_responses']}",
            f"{'='*50}",
        ]
        return "\n".join(lines)

    @staticmethod
    def comparison_table(all_metrics: Dict[str, Dict]) -> str:
        """
        Generate a comparison table across multiple models.
        
        Args:
            all_metrics: dict mapping model_name -> metrics dict
        """
        lines = [
            f"\n{'='*70}",
            f"  MODEL COMPARISON",
            f"{'='*70}",
            f"  {'Model':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}",
            f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}",
        ]
        for name, m in all_metrics.items():
            cm = m["confusion_matrix"]
            lines.append(
                f"  {name:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} "
                f"{cm['true_positives']:>5d} {cm['false_positives']:>5d} "
                f"{cm['true_negatives']:>5d} {cm['false_negatives']:>5d}"
            )
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)
