import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
from PIL import Image
import difflib
import re
import math
import json
import sys
import argparse

import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

from PaddleOCR import PaddleOCR, draw_ocr

# from VietnameseOcrCorrection.tool.predictor import Corrector
# import time
# from VietnameseOcrCorrection.tool.utils import extract_phrases

# from ultis import display_image_in_actual_size



# Specifying output path and font path.
FONT = './PaddleOCR/doc/fonts/latin.ttf'

from transformers import pipeline

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

MAX_LENGTH = 512


def predict(recognitor, detector, img_path, padding=4):
    # Load image
    img = cv2.imread(img_path)

    # Text detection
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]

    # Filter Boxes
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]

    # Add padding to boxes
    padding = 4
    for box in boxes:
        box[0][0] = box[0][0] - padding
        box[0][1] = box[0][1] - padding
        box[1][0] = box[1][0] + padding
        box[1][1] = box[1][1] + padding

    # Text recognizion
    texts = []
    for i, box in enumerate(boxes):
        try:
            # Extract cropped region
            cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            # Check if cropped image has valid dimensions
            if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid dimensions: {cropped_image.shape}")
                continue
            
            # Convert to PIL Image
            cropped_image = Image.fromarray(cropped_image)
            
            # Check PIL image dimensions
            if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                print(f"Warning: Skipping box {i} with invalid PIL dimensions: {cropped_image.size}")
                continue

            rec_result = recognitor.predict(cropped_image)
            text = rec_result#[0]

            texts.append(text)
            print(text)
            
        except Exception as e:
            print(f"Warning: Error processing box {i}: {e}")
            continue

    return boxes, texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--output', default='./runs/predict', help='path to save output file')
    parser.add_argument('--use_gpu', required=False, help='is use GPU?')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use: auto (detect automatically), cpu, or cuda')
    args = parser.parse_args()

    # Configure of VietOCR
    # Default weight
    config = Cfg.load_config_from_name('vgg_transformer')
    # Custom weight
    # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
    # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    
    # Device configuration
    if args.device == 'auto':
        # Auto-detect device: GPU if available, otherwise CPU
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            print("Auto-detected device: CUDA GPU")
        else:
            config['device'] = 'cpu'
            print("Auto-detected device: CPU")
    else:
        config['device'] = args.device
        print(f"Using specified device: {args.device}")

    recognitor = Predictor(config)

    # Config of PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=True)
    

    # Predict
    boxes, texts = predict(recognitor, detector, args.img, padding=2)

    corrections = corrector(texts, max_new_tokens=256)

    # Print predictions
    for text, pred in zip(texts, corrections):
        print("- " + pred['generated_text'])


if __name__ == "__main__":    
    main()
