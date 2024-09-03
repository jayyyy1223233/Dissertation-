import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_accuracy(train_acc_history, val_acc_history):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def display_gaf_image(gaf_path):
    gaf_img = np.load(gaf_path)
    gaf_img_normalized = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min())
    gaf_img_normalized = (gaf_img_normalized * 255).astype(np.uint8)
    gaf_pil = Image.fromarray(gaf_img_normalized)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(gaf_pil, cmap='rainbow')
    plt.axis('off')
    plt.show()
