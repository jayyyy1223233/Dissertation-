import pandas as pd
import os
from pyts.image import GramianAngularField
import cv2
import numpy as np

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def detect_patterns(data):
    morning_star_indices = []
    evening_star_indices = []
    for i in range(2, len(data) - 1):
        # Morning Star pattern
        if (data['Close'].iloc[i - 2] > data['Open'].iloc[i - 2] and 
            data['Close'].iloc[i - 1] < data['Open'].iloc[i - 1] and  
            data['Close'].iloc[i] > data['Open'].iloc[i] and  
            data['Close'].iloc[i] < data['Close'].iloc[i - 2] and 
            data['Close'].iloc[i - 1] < data['Close'].iloc[i - 2]): 
            morning_star_indices.append(i)

        # Evening Star pattern
        elif (data['Close'].iloc[i - 2] < data['Open'].iloc[i - 2] and  
              data['Close'].iloc[i - 1] > data['Open'].iloc[i - 1] and  
              data['Close'].iloc[i] < data['Open'].iloc[i] and  
              data['Close'].iloc[i] > data['Close'].iloc[i - 2] and 
              data['Close'].iloc[i - 1] > data['Close'].iloc[i - 2]):  
            evening_star_indices.append(i)

    return morning_star_indices, evening_star_indices

def save_gaf_images(df, output_dir='gaf_images'):
    os.makedirs(output_dir, exist_ok=True)
    df['YearMonth'] = df['Date'].dt.to_period('M')
    gaf = GramianAngularField(method='summation')
    
    for name, group in df.groupby('YearMonth'):
        img_resized = cv2.resize(group, (50, 50))  # Resize to match GAF input size
        img_normalized = img_resized / 255.0
        img_flattened = img_normalized.flatten().reshape(1, -1)
        gaf_img = gaf.fit_transform(img_flattened)[0]
        gaf_image_name = f'{output_dir}/gaf_{name}.npy'
        np.save(gaf_image_name, gaf_img)
        print(f'GAF image saved for {name}')

def label_data(df):
    morning_star_indices, evening_star_indices = detect_patterns(df)
    df['MorningStar'] = 0
    df['EveningStar'] = 0
    df.loc[morning_star_indices, 'MorningStar'] = 1
    df.loc[evening_star_indices, 'EveningStar'] = 1
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_labels = df.groupby('YearMonth').agg({'MorningStar': 'max', 'EveningStar': 'max'}).reset_index()
    monthly_labels['filename'] = monthly_labels['YearMonth'].astype(str).apply(lambda x: f'gaf_{x}.png.npy')
    return monthly_labels

def save_labels(labels_df, file_path='monthly_labels.csv'):
    labels_df.to_csv(file_path, index=False)
    print(f"Labels saved to {file_path}")
