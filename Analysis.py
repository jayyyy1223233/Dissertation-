import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

def predict_patterns(model, gaf_path, transform, device, threshold=0.5):
    model.eval()
    gaf_image = np.load(gaf_path)
    gaf_image = (gaf_image - gaf_image.min()) / (gaf_image.max() - gaf_image.min())
    gaf_image = (gaf_image * 255).astype(np.uint8)
    gaf_pil = Image.fromarray(gaf_image)
    if transform:
        gaf_pil = transform(gaf_pil)
    gaf_pil = gaf_pil.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(gaf_pil)
        probs = torch.sigmoid(output)
        preds = probs > threshold
    return preds.cpu().numpy()[0], probs.cpu().numpy()[0]

def predict_patterns_and_compare(model, gaf_path, transform, device, labels_csv, threshold=0.5):
    monthly_labels = pd.read_csv(labels_csv)
    gaf_filename = gaf_path.split('/')[-1]
    actual_row = monthly_labels[monthly_labels['filename'] == gaf_filename]

    if actual_row.empty:
        print(f"No actual data found for {gaf_filename}.")
        return None, None

    actual_morning_star = actual_row['MorningStar'].values[0]
    actual_evening_star = actual_row['EveningStar'].values[0]
    actual_prediction = actual_row['Prediction'].values[0]
    preds, probs = predict_patterns(model, gaf_path, transform, device, threshold)
    
    predicted_pattern = "Bullish" if probs[0] > probs[1] else "Bearish"
    return predicted_pattern, actual_prediction

def analyze_detected_vs_actual_patterns(model, labels_csv, transform, device):
    monthly_labels = pd.read_csv(labels_csv)
    detected_patterns = []
    actual_patterns = []
    
    for index, row in monthly_labels.iterrows():
        gaf_path = f"gaf_images/{row['filename']}"
        print(f"Testing {gaf_path}...")
        detected_pattern, actual_pattern = predict_patterns_and_compare(model, gaf_path, transform, device, labels_csv)
        if detected_pattern is not None and actual_pattern is not None:
            detected_patterns.append(detected_pattern)
            actual_patterns.append(actual_pattern)
    
    detected_counter = Counter(detected_patterns)
    actual_counter = Counter(actual_patterns)
    
    labels = ['Bullish', 'Bearish']
    detected_counts = [detected_counter.get('Bullish', 0), detected_counter.get('Bearish', 0)]
    actual_counts = [actual_counter.get('Bullish', 0), actual_counter.get('Bearish', 0)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, detected_counts, width, label='Detected')
    rects2 = ax.bar(x + width/2, actual_counts, width, label='Actual')
    
    ax.set_xlabel('Pattern')
    ax.set_ylabel('Counts')
    ax.set_title('Detected vs. Actual Patterns')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.show()
