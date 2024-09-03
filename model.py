import torch
import numpy as np
from sklearn.metrics import classification_report
from torchsummary import summary

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    return all_labels, all_preds

def print_classification_report(all_labels, all_preds, target_names):
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df[['precision', 'recall', 'f1-score']])

def model_summary(model, input_size):
    summary(model, input_size=input_size)
