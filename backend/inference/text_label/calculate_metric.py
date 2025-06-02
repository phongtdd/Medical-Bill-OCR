import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from text_label.get_label import predict_label


def get_p_r_f(data):
    df = pd.read_json(f"{data}/data_CV.json")

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]
        predicted_label = predict_label(text)

        y_true.append(true_label)
        y_pred.append(predicted_label)

    unique_labels = sorted(set(y_true + y_pred))

    report = classification_report(
        y_true, y_pred, labels=unique_labels, output_dict=False, digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(xticks_rotation=45)
    # plt.tight_layout()
    # plt.show()

    print(
        f"\nPrecision: {precision_score(y_true, y_pred, average='weighted'):.4f}",
        f"\nRecall: {recall_score(y_true, y_pred, average='weighted'):.4f}",
        f"\nF1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}",
    )


if __name__ == "__main__":
    data = "data"
    get_p_r_f(data)
