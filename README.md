# Vietnamese OCR with Text Correction

🇻🇳 **Vietnamese OCR with intelligent text correction capabilities**

This project combines state-of-the-art OCR frameworks with AI-powered text correction to deliver highly accurate Vietnamese text extraction from images. The system integrates PaddleOCR for text detection, VietOCR for recognition, and a Transformer-based model for post-processing text correction.

## ✨ Features

-  **Text Detection**: Advanced text localization using PaddleOCR's DB algorithm
-  **Text Recognition**: High-accuracy Vietnamese text recognition with VietOCR
-  **Text Correction**: AI-powered spelling and grammar correction using `bmd1905/vietnamese-correction-v2`
-  **GPU Acceleration**: CUDA support for faster processing
-  **Robust Error Handling**: Gracefully handles problematic image regions
-  **Flexible Device Support**: Auto-detection or manual selection of CPU/GPU
-  **Multi-format Output**: Both raw OCR and corrected text results

##  Architecture

### 1. Text Detection
- **Algorithm**: DB (Differentiable Binarization) from PaddleOCR
- **Features**: High-speed text region localization with padding optimization
- **Output**: Precise bounding boxes around text regions

### 2. Text Recognition  
- **Framework**: VietOCR with Transformer architecture
- **Model**: VGG + Transformer with beam search
- **Specialization**: Optimized for Vietnamese text patterns and diacritics

### 3. Text Correction
- **Model**: `bmd1905/vietnamese-correction-v2` (T5-based)
- **Capabilities**: 
  - Spelling correction
  - Grammar enhancement
  - Proper capitalization
  - Punctuation normalization

##  Quick Start

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for acceleration)
- Conda or pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bmd1905/vietnamese-ocr
```

2. **Create and activate conda environment**
```bash
# Option 1: Full environment reproduction
conda env create -f environment.yml
conda activate ocr

# Option 2: Manual setup
conda create -n ocr python=3.11.9
conda activate ocr
pip install -r requirements.txt
```

### Usage

#### Command Line Interface

**Basic usage (auto-detect device):**
```bash
python predict.py --img path/to/image.jpg --output ./output/
```

**Force GPU acceleration:**
```bash
python predict.py --img path/to/image.jpg --output ./output/ --device cuda
```

**Force CPU processing:**
```bash
python predict.py --img path/to/image.jpg --output ./output/ --device cpu
```

#### Jupyter Notebook
Explore the interactive notebook: [`predict.ipynb`](predict.ipynb)

### Example Output

```bash
# Raw OCR Output:
XE 3 BÁNH VÀ 4 BANH

# Corrected Output:
- XE 3 BÁNH VÀ 4 BÁNH.
```

## 🔧 Configuration

### Device Selection
- `--device auto`: Automatically detect and use GPU if available
- `--device cuda`: Force CUDA GPU usage  
- `--device cpu`: Force CPU processing

### Model Configuration
The system uses pre-trained models:
- **Detection**: PaddleOCR PP-OCRv3
- **Recognition**: VietOCR VGG-Transformer
- **Correction**: `bmd1905/vietnamese-correction-v2`

## 📚 References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Text detection framework
- [VietOCR](https://github.com/pbcquoc/vietocr) - Vietnamese text recognition
- [Transformers](https://huggingface.co/transformers/) - Text correction models
- [Vietnamese Correction Model](https://huggingface.co/bmd1905/vietnamese-correction-v2)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
