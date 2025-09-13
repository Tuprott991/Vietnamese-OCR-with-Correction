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
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

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
        # Import required functions from VietOCR
        from vietocr.vietocr.tool.translate import process_input, translate, translate_beam_search
        self.process_input = process_input
        self.translate = translate
        self.translate_beam_search = translate_beam_search
        
        # Store image processing parameters
        self.image_height = config['dataset']['image_height']
        self.image_min_width = config['dataset']['image_min_width'] 
        self.image_max_width = config['dataset']['image_max_width']
        self.beamsearch = config['predictor']['beamsearch']
        
        print("✅ BatchPredictor initialized successfully with VietOCR functions")
    
    def predict_batch(self, images, batch_size=8):
        """
        Batch prediction for multiple PIL images using VietOCR's built-in batch processing
        Args:
            images: List of PIL Images
            batch_size: Internal batch size for processing
        Returns:
            List of predicted text strings
        """
        if not images:
            return []
        
        # Check if we have the necessary attributes for batch processing
        if not hasattr(self, 'model') or not hasattr(self, 'process_input'):
            print("Missing required attributes for batch processing, falling back to sequential")
            results = []
            for img in images:
                try:
                    result = self.predict(img)
                    results.append(result)
                except:
                    results.append("")
            return results
        
        # Use VietOCR's built-in predict_batch method if available
        if hasattr(super(), 'predict_batch'):
            try:
                print(f"🚀 Using VietOCR's built-in batch processing for {len(images)} images")
                return super().predict_batch(images)
            except Exception as e:
                print(f"VietOCR batch processing error: {e}, falling back to custom implementation")
        
        # Custom batch processing with proper tensor handling
        results = []
        
        # Group images by similar aspect ratios for better batching
        from collections import defaultdict
        
        # Group images by width (after preprocessing) to avoid tensor size mismatch
        width_groups = defaultdict(list)
        processed_images = []
        
        # First pass: process all images and group by width
        for idx, img in enumerate(images):
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Process input to get tensor
                img_tensor = self.process_input(img, self.image_height, 
                                              self.image_min_width, self.image_max_width)
                width = img_tensor.shape[-1]  # Get width dimension
                width_groups[width].append((idx, img_tensor))
                processed_images.append(None)  # Placeholder
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                processed_images.append("")
        
        # Process each width group in batches
        for width, img_group in width_groups.items():
            group_indices = [item[0] for item in img_group]
            group_tensors = [item[1] for item in img_group]
            
            # Process this width group in batches
            for i in range(0, len(group_tensors), batch_size):
                batch_tensors = group_tensors[i:i + batch_size]
                batch_indices = group_indices[i:i + batch_size]
                
                try:
                    if batch_tensors:
                        # Now all tensors in this batch have the same width, so we can stack them
                        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
                        
                        # Run batch inference
                        with torch.no_grad():
                            if self.beamsearch:
                                # For beam search, process individually
                                batch_results = []
                                for j in range(batch_tensor.size(0)):
                                    single_img = batch_tensor[j:j+1]
                                    sent = self.translate_beam_search(single_img, self.model)
                                    decoded_sent = self.vocab.decode(sent)
                                    batch_results.append(decoded_sent)
                            else:
                                # For regular translation
                                batch_output, _ = self.translate(batch_tensor, self.model)
                                batch_results = []
                                for output in batch_output:
                                    s = output.tolist()
                                    decoded_sent = self.vocab.decode(s)
                                    batch_results.append(decoded_sent)
                        
                        # Store results in correct positions
                        for idx, result in zip(batch_indices, batch_results):
                            processed_images[idx] = result
                            
                except Exception as e:
                    print(f"Batch processing error for width {width}: {e}")
                    # Fallback to individual processing for this batch
                    for idx in batch_indices:
                        try:
                            original_img = images[idx]
                            result = self.predict(original_img)
                            processed_images[idx] = result
                        except:
                            processed_images[idx] = ""
        
        # Fill any remaining None values with empty strings
        results = [result if result is not None else "" for result in processed_images]
        
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


def load_image_only(img_path):
    """Helper function for parallel image loading only (safe)"""
    try:
        start_time = time.time()
        img = cv2.imread(img_path)
        load_time = time.time() - start_time
        
        if img is None:
            return None, f"Could not load image {img_path}", load_time
        
        return (img, img_path), None, load_time
        
    except Exception as e:
        return None, f"Error loading {img_path}: {e}", 0


def detect_image_sequential(detector, img, img_path):
    """Sequential detection function - thread safe"""
    try:
        detect_start = time.time()
        result = detector.ocr(img_path, cls=False, det=True, rec=False)
        detect_time = time.time() - detect_start
        
        if not result or not result[0]:
            return None, f"No text detected in {img_path}", detect_time
            
        result = result[0]
        
        # Filter Boxes
        boxes = []
        for line in result:
            boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
        boxes = boxes[::-1]
        
        return boxes, None, detect_time
        
    except Exception as e:
        return None, f"Error detecting {img_path}: {e}", 0


def predict_batch_true(recognitor, detector, image_paths, recognition_batch_size=8, padding=4, use_parallel_io=True):
    """
    Process a batch of images for OCR with true batch processing for recognition
    """
    results = {}
    total_load_time = 0
    total_detect_time = 0
    
    print(f"🔄 Loading and detecting {len(image_paths)} images...")
    start_time = time.time()
    
    # Load all images and perform detection
    batch_images = []
    valid_paths = []
    all_boxes = []
    
    if use_parallel_io and len(image_paths) > 2:
        # Parallel I/O processing (only for loading images - safer)
        # Get max_workers from the stored attribute
        max_workers_setting = getattr(process_video_folder, '_max_workers', 0)
        
        if max_workers_setting == 0:
            # Auto-detect optimal worker count
            cpu_count = multiprocessing.cpu_count()
            # For Kaggle/high-performance systems, use more workers
            max_workers = min(12, max(4, cpu_count - 2))  # Leave 2 CPUs for other tasks
        else:
            max_workers = max_workers_setting
            
        print(f"📈 Using {max_workers} parallel workers for image loading (CPU cores: {multiprocessing.cpu_count()})")
        
        # Step 1: Parallel image loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            load_results = list(executor.map(load_image_only, image_paths))
        
        # Step 2: Sequential detection (PaddleOCR is not thread-safe)
        for i, (load_result, load_error, load_time) in enumerate(load_results):
            total_load_time += load_time
            
            if load_result is None:
                if load_error:
                    print(f"Warning: {load_error}")
                results[os.path.basename(image_paths[i])] = []
                continue
            
            img, img_path = load_result
            
            # Sequential detection to avoid segfaults
            boxes, detect_error, detect_time = detect_image_sequential(detector, img, img_path)
            total_detect_time += detect_time
            
            if boxes is None:
                if detect_error:
                    print(f"Warning: {detect_error}")
                results[os.path.basename(img_path)] = []
                continue
            
            batch_images.append(img)
            valid_paths.append(img_path)
            all_boxes.append(boxes)
    else:
        # Sequential processing (safest method)
        for img_path in image_paths:
            # Load image
            load_result, load_error, load_time = load_image_only(img_path)
            total_load_time += load_time
            
            if load_result is None:
                if load_error:
                    print(f"Warning: {load_error}")
                results[os.path.basename(img_path)] = []
                continue
            
            img, img_path = load_result
            
            # Sequential detection
            boxes, detect_error, detect_time = detect_image_sequential(detector, img, img_path)
            total_detect_time += detect_time
            
            if boxes is None:
                if detect_error:
                    print(f"Warning: {detect_error}")
                results[os.path.basename(img_path)] = []
                continue
            
            batch_images.append(img)
            valid_paths.append(img_path)
            all_boxes.append(boxes)
    
    io_time = time.time() - start_time
    print(f"⏱️  I/O Phase: {io_time:.2f}s (Load: {total_load_time:.2f}s, Detect: {total_detect_time:.2f}s)")
    
    if not batch_images:
        return results
    
    if not batch_images:
        return results
    
    # Collect all cropped images for batch recognition (GPU bottleneck starts here)
    print(f"🖼️  Cropping {sum(len(boxes) for boxes in all_boxes)} text regions...")
    crop_start = time.time()
    
    all_cropped_images = []
    image_indices = []
    
    for img_idx, (img, boxes) in enumerate(zip(batch_images, all_boxes)):
        for box in boxes:
            # Add padding
            box[0][0] = max(0, box[0][0] - padding)
            box[0][1] = max(0, box[0][1] - padding)
            box[1][0] = min(img.shape[1], box[1][0] + padding)
            box[1][1] = min(img.shape[0], box[1][1] + padding)
            
            # Extract cropped region (CPU operation)
            cropped = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                if cropped_pil.size[0] > 0 and cropped_pil.size[1] > 0:
                    all_cropped_images.append(cropped_pil)
                    image_indices.append(img_idx)
    
    crop_time = time.time() - crop_start
    print(f"⏱️  Cropping Phase: {crop_time:.2f}s")
    
    # Batch recognition - This is where GPU should work hard
    batch_texts = []
    if all_cropped_images:
        try:
            print(f"🚀 GPU Processing {len(all_cropped_images)} text regions in batches of {recognition_batch_size}")
            gpu_start = time.time()
            batch_texts = recognitor.predict_batch(all_cropped_images, recognition_batch_size)
            gpu_time = time.time() - gpu_start
            print(f"⏱️  GPU Recognition Phase: {gpu_time:.2f}s ({len(all_cropped_images)/gpu_time:.2f} texts/sec)")
        except Exception as e:
            print(f"Batch recognition error: {e}")
            print("Falling back to sequential recognition...")
            # Fallback to sequential processing
            fallback_start = time.time()
            for cropped_img in all_cropped_images:
                try:
                    text = recognitor.predict(cropped_img)
                    batch_texts.append(text)
                except:
                    batch_texts.append("")
            fallback_time = time.time() - fallback_start
            print(f"⏱️  Fallback Recognition Phase: {fallback_time:.2f}s")
    
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
            rec, det, paths, recognition_batch_size, 4, 
            getattr(process_video_folder, '_parallel_io', False)
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
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for processing images (default: 16)')
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
    parser.add_argument('--recognition_batch_size', type=int, default=32, 
                       help='Batch size for text recognition (default: 32 for batch processing mode)')
    parser.add_argument('--aggressive_batching', action='store_true', 
                       help='Use very large batch sizes for maximum GPU utilization (requires 8GB+ GPU memory)')
    parser.add_argument('--parallel_io', action='store_true', 
                       help='Use parallel I/O processing to reduce CPU bottlenecks (recommended)')
    parser.add_argument('--profile_performance', action='store_true', 
                       help='Show detailed performance profiling information')
    parser.add_argument('--max_workers', type=int, default=0, 
                       help='Maximum parallel workers for I/O operations (0=auto, default: 8 for Kaggle)')
    
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

    # Adjust batch sizes based on aggressive batching flag
    if args.aggressive_batching:
        args.batch_size = max(args.batch_size, 64)
        args.recognition_batch_size = max(args.recognition_batch_size, 128)
        print(f"🚀 Aggressive batching enabled: batch_size={args.batch_size}, recognition_batch_size={args.recognition_batch_size}")

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
    
    # Store parallel I/O settings for process_video_folder
    process_video_folder._parallel_io = args.parallel_io
    process_video_folder._max_workers = args.max_workers
    
    # Print resource utilization recommendations
    print(f"\n📊 Current batch configuration:")
    print(f"   - Image batch size: {args.batch_size}")
    print(f"   - Recognition batch size: {args.recognition_batch_size}")
    print(f"   - Parallel I/O: {'Enabled' if args.parallel_io else 'Disabled'}")
    print(f"   - Max workers: {args.max_workers if args.max_workers > 0 else 'Auto (8-12)'}")
    print(f"   - Available CPU cores: {multiprocessing.cpu_count()}")
    print(f"   - GPU memory available: ~15GB")
    print(f"   - Performance profiling: {'Enabled' if args.profile_performance else 'Disabled'}")
    
    if args.profile_performance:
        print(f"\n🎯 CPU Usage Analysis:")
        print(f"   - Image Loading: ~30-40% of CPU time")
        print(f"   - PaddleOCR Detection: ~40-50% of CPU time") 
        print(f"   - Image Cropping: ~10-15% of CPU time")
        print(f"   - GPU Recognition: Should be GPU-bound, not CPU-bound")
        print(f"   - Recommendation: Use --parallel_io to reduce CPU bottlenecks")

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