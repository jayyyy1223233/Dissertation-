# Candlestick Pattern Recognition using Deep Learning

## Overview
This repository contains the code and resources for recognizing and categorizing Morning Star and Evening Star candlestick patterns using deep learning models.

## Project Structure
- `preprocessing.py`: Contains functions for data preprocessing, pattern detection, and GAF image generation.
- `resnet.py`: Defines and trains a ResNet-based model.
- `cnn.py`: Defines and trains a custom CNN model.
- `model.py`: General functions for model evaluation, prediction, and summary.
- `plotter.py`: Handles the plotting of accuracy graphs and visualization of GAF images.
- `analysis.py`: Performs analysis and comparisons between detected and actual patterns.

## Requirements
The required Python packages can be found in `requirements.txt`.

## Getting Started
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Use the scripts to preprocess data, train models, and analyze results.

## Usage
Each script is designed to be modular. You can run them independently or integrate them as needed.

## Results
Results will be stored in the `results/` directory, including plots, saved models, and analysis outputs.

## Acknowledgments
This project was developed to explore the use of deep learning in financial analytics, specifically for candlestick pattern recognition.

