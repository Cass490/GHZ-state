import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def evaluate(y_true, y_pred, scores=None):
    result = {name: float(fn(y_true, y_pred, **({'zero_division': 0} if name != 'accuracy' else {})))
              for name, fn in [('accuracy', accuracy_score), ('precision', precision_score),
                               ('recall', recall_score), ('f1', f1_score)]}
    result['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    result['roc_auc'] = float(roc_auc_score(y_true, scores)) if scores is not None and len(set(y_true)) == 2 else None
    result['num_samples'] = len(y_true)
    return result


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
