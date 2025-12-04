from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import pandas as pd

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted")
    }

def get_classification_report_df(y_true, y_pred):
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    return report_df

def get_confusion_matrix_df(y_true, y_pred, labels=["negative","neutral","positive"]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df
