import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import json
import sys
import argparse
import glob
from pathlib import Path

import torch

import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

from PaddleOCR import PaddleOCR


class BatchPredictor(Predictor):
    """
    Extended VietOCR Predictor with batch processing capabilities
    """
    def __init__(self, config):
        super().__init__(config)
    
    def predict_batch(self, images, batch_size=8):
        """
        Batch prediction for multiple PIL images
        Args:
            images: List of PIL Images
            batch_size: Internal batch size for processing
        Returns:
            List of predicted text strings
        """
        if not images:
            return []
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            
            try:
                # Convert PIL images to tensors
                batch_tensors = []
                for img in batch_imgs:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_tensor = self.transforms(img)
                    batch_tensors.append(img_tensor)
                
                # Stack into batch tensor
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    
                    # Run batch inference
                    with torch.no_grad():
                        batch_output = self.model(batch_tensor)
                        
                        # Process batch output
                        for j in range(batch_output.size(0)):
                            single_output = batch_output[j:j+1]  # Keep batch dimension
                            if self.beamsearch:
                                sent = self.translate_beam_search(single_output)[0]
                            else:
                                sent = self.translate(single_output)[0]
                            results.append(sent)
                        
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Fallback to individual processing for this batch
                for img in batch_imgs:
                    try:
                        result = self.predict(img)
                        results.append(result)
                    except:
                        results.append("")
        
        return results


def predict_batch_sequential(recognitor, detector, image_paths, padding=4):
    """
    Process a batch of images for OCR (Sequential processing - original method)
    """
    results = {}
    
    for img_path in image_paths:
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                results[os.path.basename(img_path)] = []
                continue

            # Text detection
            result = detector.ocr(img_path, cls=False, det=True, rec=False)
            if not result or not result[0]:
                print(f"Warning: No text detected in {img_path}")
                results[os.path.basename(img_path)] = []
                continue
                
            result = result[0]

            # Filter Boxes
            boxes = []
            for line in result:
                boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
            boxes = boxes[::-1]

            # Add padding to boxes
            for box in boxes:
                box[0][0] = max(0, box[0][0] - padding)
                box[0][1] = max(0, box[0][1] - padding)
                box[1][0] = min(img.shape[1], box[1][0] + padding)
                box[1][1] = min(img.shape[0], box[1][1] + padding)

            # Text recognition
            texts = []
            for i, box in enumerate(boxes):
                try:
                    # Extract cropped region
                    cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                    
                    # Check if cropped image has valid dimensions
                    if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                        print(f"Warning: Skipping box {i} in {img_path} with invalid dimensions: {cropped_image.shape}")
                        continue
                    
                    # Convert to PIL Image
                    cropped_image = Image.fromarray(cropped_image)
                    
                    # Check PIL image dimensions
                    if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                        print(f"Warning: Skipping box {i} in {img_path} with invalid PIL dimensions: {cropped_image.size}")
                        continue

                    rec_result = recognitor.predict(cropped_image)
                    text = rec_result
                    
                    if text.strip():  # Only add non-empty text
                        texts.append(text.strip())
                    
                except Exception as e:
                    print(f"Warning: Error processing box {i} in {img_path}: {e}")
                    continue

            results[os.path.basename(img_path)] = texts
            print(f"Processed {os.path.basename(img_path)}: {len(texts)} text regions found")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[os.path.basename(img_path)] = []
            continue

    return results


def predict_batch_true(recognitor, detector, image_paths, recognition_batch_size=8, padding=4):
    """
    Process a batch of images for OCR with true batch processing for recognition
    """
    results = {}
    
    # Load all images and perform detection
    batch_images = []
    valid_paths = []
    all_boxes = []
    
    for img_path in image_paths:
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                results[os.path.basename(img_path)] = []
                continue

            # Text detection (still sequential as PaddleOCR doesn't support batch detection well)
            result = detector.ocr(img_path, cls=False, det=True, rec=False)
            if not result or not result[0]:
                print(f"Warning: No text detected in {img_path}")
                results[os.path.basename(img_path)] = []
                continue
                
            result = result[0]

            # Filter Boxes
            boxes = []
            for line in result:
                boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
            boxes = boxes[::-1]

            batch_images.append(img)
            valid_paths.append(img_path)
            all_boxes.append(boxes)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[os.path.basename(img_path)] = []
            continue
    
    if not batch_images:
        return results
    
    # Collect all cropped images for batch recognition
    all_cropped_images = []
    image_indices = []
    
    for img_idx, (img, boxes) in enumerate(zip(batch_images, all_boxes)):
        for box in boxes:
            # Add padding
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(img.shape[1], box[1][0] + padding)
            box[1][1] = min(img.shape[0], box[1][1] + padding)
            
            # Extract cropped region
            cropped = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                if cropped_pil.size[0] > 0 and cropped_pil.size[1] > 0:
                    all_cropped_images.append(cropped_pil)
                    image_indices.append(img_idx)
    
    # Batch recognition - This is where we get the real speedup
    batch_texts = []
    if all_cropped_images:
        try:
            print(f"Processing {len(all_cropped_images)} text regions in batches of {recognition_batch_size}")
            batch_texts = recognitor.predict_batch(all_cropped_images, recognition_batch_size)
        except Exception as e:
            print(f"Batch recognition error: {e}")
            print("Falling back to sequential recognition...")
            # Fallback to sequential processing
            for cropped_img in all_cropped_images:
                try:
                    text = recognitor.predict(cropped_img)
                    batch_texts.append(text)
                except:
                    batch_texts.append("")
    
    # Group results back by original image
    text_idx = 0
    for img_idx in range(len(valid_paths)):
        img_path = valid_paths[img_idx]
        texts = []
        
        while text_idx < len(image_indices) and image_indices[text_idx] == img_idx:
            if text_idx < len(batch_texts) and batch_texts[text_idx].strip():
                texts.append(batch_texts[text_idx].strip())
            text_idx += 1
            
        results[os.path.basename(img_path)] = texts
        print(f"Processed {os.path.basename(img_path)}: {len(texts)} text regions found")
    
    return results


def process_video_folder(recognitor, detector, video_folder_path, output_dir, batch_size=8, use_batch_processing=False, recognition_batch_size=8):
    """
    Process all images in a video folder and save results to JSON
    """
    video_name = os.path.basename(video_folder_path)
    print(f"Processing folder: {video_name}")
    
    # Get all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(video_folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(video_folder_path, ext.upper())))
    
    image_paths.sort()  # Sort to ensure consistent ordering
    
    if not image_paths:
        print(f"Warning: No images found in {video_folder_path}")
        return
    
    print(f"Found {len(image_paths)} images in {video_name}")
    
    # Select the appropriate prediction function
    if use_batch_processing:
        predict_function = lambda rec, det, paths: predict_batch_true(
            rec, det, paths, recognition_batch_size
        )
    else:
        predict_function = predict_batch_sequential
    
    # Process images in batches
    all_results = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        batch_results = predict_function(recognitor, detector, batch_paths)
        all_results.update(batch_results)
    
    # Save results to JSON
    output_file = os.path.join(output_dir, f"{video_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Process Kaggle image dataset for Vietnamese OCR')
    parser.add_argument('--input_path', required=True, 
                       help='Input path like /kaggle/input/dataset_batch_2/L22/L22/')
    parser.add_argument('--output_dir', default='./kaggle_output', 
                       help='Output directory for JSON files')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for processing images')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use: auto (detect automatically), cpu, or cuda')
    parser.add_argument('--start_video', type=int, default=1, 
                       help='Starting video number (default: 1)')
    parser.add_argument('--end_video', type=int, default=99, 
                       help='Ending video number (default: 99)')
    parser.add_argument('--folder_prefix', default='L22_V', 
                       help='Folder prefix pattern (default: L22_V for L22_V001, L22_V002, etc.)')
    parser.add_argument('--auto_detect', action='store_true', 
                       help='Auto-detect all video folders in the input path instead of using number range')
    parser.add_argument('--use_batch_processing', action='store_true', 
                       help='Use true batch processing for recognition (faster but uses more memory)')
    parser.add_argument('--recognition_batch_size', type=int, default=8, 
                       help='Batch size for text recognition (only for batch processing mode)')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')
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

    # Use BatchPredictor if batch processing is enabled
    if args.use_batch_processing:
        recognitor = BatchPredictor(config)
        print(f"Using BatchPredictor for true batch processing with recognition batch size: {args.recognition_batch_size}")
    else:
        recognitor = Predictor(config)
        print("Using standard Predictor for sequential processing")

    # Configure PaddleOCR
    use_gpu = args.device in ['cuda', 'auto'] and torch.cuda.is_available()
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=use_gpu)
    print(f"PaddleOCR using GPU: {use_gpu}")

    # Process video folders
    processed_count = 0
    failed_count = 0
    
    if args.auto_detect:
        # Auto-detect all folders in the input path
        print(f"Auto-detecting folders in: {args.input_path}")
        
        if not os.path.exists(args.input_path):
            print(f"Error: Input path {args.input_path} does not exist!")
            return
        
        # Get all subdirectories
        all_folders = [f for f in os.listdir(args.input_path) 
                      if os.path.isdir(os.path.join(args.input_path, f))]
        all_folders.sort()
        
        print(f"Found {len(all_folders)} folders: {all_folders}")
        
        for folder_name in all_folders:
            video_folder_path = os.path.join(args.input_path, folder_name)
            
            try:
                output_file = process_video_folder(recognitor, detector, video_folder_path, 
                                                 args.output_dir, args.batch_size, 
                                                 args.use_batch_processing, args.recognition_batch_size)
                if output_file:
                    processed_count += 1
                    print(f"✓ Successfully processed {folder_name}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {folder_name}")
            
            except Exception as e:
                print(f"✗ Error processing {folder_name}: {e}")
                failed_count += 1
                continue
    
    else:
        # Use the traditional numbered folder approach
        for video_num in range(args.start_video, args.end_video + 1):
            video_folder_name = f"{args.folder_prefix}{video_num:03d}"
            video_folder_path = os.path.join(args.input_path, video_folder_name)
            
            if not os.path.exists(video_folder_path):
                print(f"Warning: Folder {video_folder_path} does not exist, skipping...")
                failed_count += 1
                continue
            
            try:
                output_file = process_video_folder(recognitor, detector, video_folder_path, 
                                                 args.output_dir, args.batch_size, 
                                                 args.use_batch_processing, args.recognition_batch_size)
                if output_file:
                    processed_count += 1
                    print(f"✓ Successfully processed {video_folder_name}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {video_folder_name}")
            
            except Exception as e:
                print(f"✗ Error processing {video_folder_name}: {e}")
                failed_count += 1
                continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {processed_count} folders")
    print(f"Failed: {failed_count} folders")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":    
    main()