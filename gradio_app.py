import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR

# Initialize models
def initialize_models():
    """Initialize OCR and correction models"""
    # Configure VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    
    # Auto-detect device
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print("Using CUDA GPU")
    else:
        config['device'] = 'cpu'
        print("Using CPU")

    recognitor = Predictor(config)
    
    # Configure PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=torch.cuda.is_available())
    
    # Initialize correction model
    corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
    
    return recognitor, detector, corrector

# Global model variables
recognitor, detector, corrector = initialize_models()

def predict_ocr(image, padding=4):
    """
    Perform OCR on the uploaded image
    
    Args:
        image: PIL Image object
        padding: padding around detected text boxes
        
    Returns:
        tuple: (boxes, texts) - detected boxes and recognized texts
    """
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    # Save temporary image for PaddleOCR
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img)
    
    # Text detection using PaddleOCR
    result = detector.ocr(temp_path, cls=False, det=True, rec=False)
    result = result[:][:][0]
    
    if not result:
        return [], []
    
    # Filter and process boxes
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
                continue
            
            # Convert to PIL Image
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            
            # Check PIL image dimensions
            if cropped_image.size[0] <= 0 or cropped_image.size[1] <= 0:
                continue

            # Recognize text
            rec_result = recognitor.predict(cropped_image)
            texts.append(rec_result)
            
        except Exception as e:
            print(f"Warning: Error processing box {i}: {e}")
            continue
    
    return boxes, texts

def correct_texts(texts):
    """
    Apply text correction to the recognized texts
    
    Args:
        texts: list of recognized text strings
        
    Returns:
        list: corrected text strings
    """
    if not texts:
        return []
    
    try:
        corrections = corrector(texts, max_new_tokens=256)
        corrected_texts = []
        
        for correction in corrections:
            if isinstance(correction, dict) and 'generated_text' in correction:
                corrected_texts.append(correction['generated_text'])
            else:
                corrected_texts.append(str(correction))
                
        return corrected_texts
    except Exception as e:
        print(f"Error in correction: {e}")
        return texts  # Return original texts if correction fails

def process_image(image):
    """
    Main processing function for the Gradio interface
    
    Args:
        image: PIL Image uploaded by user
        
    Returns:
        tuple: (raw_text, corrected_text) - OCR results and corrected results
    """
    if image is None:
        return "Please upload an image.", "Please upload an image."
    
    try:
        # Perform OCR
        boxes, texts = predict_ocr(image)
        
        if not texts:
            return "No text detected in the image.", "No text detected in the image."
        
        # Join all detected texts
        raw_text = "\n".join(texts)
        
        # Apply correction
        corrected_texts = correct_texts(texts)
        corrected_text = "\n".join(corrected_texts)
        
        return raw_text, corrected_text
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return error_msg, error_msg

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .image-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Vietnamese OCR with Correction") as interface:
        gr.Markdown(
            """
            # 🇻🇳 Vietnamese OCR with Text Correction
            
            Upload an image containing Vietnamese text to extract and correct the text automatically.
            The system uses PaddleOCR for text detection and VietOCR for text recognition, 
            followed by an AI-powered correction model to improve accuracy.
            """,
            elem_classes="title"
        )
        
        with gr.Row():
            # Left column - Image upload
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    type="pil",
                    label="Upload an image with Vietnamese text",
                    elem_classes="image-container"
                )
                
                process_btn = gr.Button(
                    "🔍 Extract & Correct Text",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    **Tips:**
                    - Use clear, high-resolution images
                    - Ensure good lighting and contrast
                    - Avoid blurry or tilted text
                    - Supported formats: JPG, PNG, etc.
                    """
                )
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Results")
                
                with gr.Tab("Raw OCR"):
                    raw_output = gr.Textbox(
                        label="Extracted Text (Raw OCR)",
                        placeholder="Raw OCR results will appear here...",
                        lines=8,
                        max_lines=15,
                        elem_classes="result-container"
                    )
                
                with gr.Tab("Corrected Text"):
                    corrected_output = gr.Textbox(
                        label="Corrected Text",
                        placeholder="Corrected text will appear here...",
                        lines=8,
                        max_lines=15,
                        elem_classes="result-container"
                    )
                
                # Copy buttons
                with gr.Row():
                    copy_raw_btn = gr.Button("📋 Copy Raw Text", size="sm")
                    copy_corrected_btn = gr.Button("📋 Copy Corrected Text", size="sm")
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[raw_output, corrected_output],
            show_progress=True
        )
        
        # Auto-process when image is uploaded
        image_input.change(
            fn=process_image,
            inputs=[image_input],
            outputs=[raw_output, corrected_output],
            show_progress=True
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #7f8c8d;">
                <p>Powered by PaddleOCR, VietOCR, and Transformers | Built with Gradio</p>
            </div>
            """,
            elem_classes="footer"
        )
    
    return interface

def main():
    """Launch the Gradio application"""
    print("Initializing Vietnamese OCR with Correction Interface...")
    print("Models loaded successfully!")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
