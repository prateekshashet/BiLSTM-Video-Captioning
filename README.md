# BiLSTM-Video-Captioning
A deep learning model using Bidirectional LSTM networks to analyze video sequences and automatically generate meaningful textual descriptions of the content
# BiLSTM-Based Video Captioning System

## Overview

This project presents a deep learning-based video captioning system that generates natural language descriptions from video sequences. The proposed architecture combines CNN-based feature extraction, YOLOv8 object detection, BiLSTM temporal encoding, Transformer-based decoding, and BLIP vocabulary refinement to produce semantically rich captions.

The system is capable of generating:

* Broad descriptive captions
* Concise summary captions

---

## Features

* Frame extraction and preprocessing
* CNN-based spatial feature extraction
* YOLOv8 object detection for ROI grounding
* BiLSTM temporal sequence modeling
* Transformer encoder-decoder architecture
* BLIP-based vocabulary enhancement
* Broad and summary caption generation
* Evaluation using BLEU, METEOR, ROUGE-L, and CIDEr metrics

---

## System Architecture

Input Video
→ Frame Extraction
→ CNN Feature Extraction
→ YOLOv8 Object Detection
→ Feature Fusion
→ BiLSTM Encoder
→ Transformer Encoder
→ Hierarchical Decoder
→ BLIP Vocabulary Refinement
→ Final Caption Output

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* YOLOv8
* BiLSTM
* Transformer Networks
* BLIP
* NumPy
* Pandas

---

## Dataset

The model uses the MSR-VTT dataset for training and evaluation.

Dataset Features:

* 10,000 video clips
* Multiple video categories
* 20 human-generated captions per video

---

## Model Components

### 1. Frame Extraction

Videos are uniformly sampled into key frames for temporal consistency.

### 2. Feature Extraction

CNN (ResNet-based) features are extracted from each frame.

### 3. Object Detection

YOLOv8 identifies objects and extracts region-of-interest (ROI) features.

### 4. Temporal Encoding

BiLSTM captures both past and future temporal dependencies across frames.

### 5. Caption Generation

Transformer decoder generates natural language captions.

### 6. Language Refinement

BLIP improves vocabulary alignment and caption fluency.

---

## Evaluation Metrics

The system is evaluated using:

* BLEU Score
* METEOR
* ROUGE-L
* CIDEr

### Best Results

| Metric  | Summary Caption Score |
| ------- | --------------------- |
| BLEU-4  | 0.8172                |
| METEOR  | 0.8928                |
| ROUGE-L | 0.8909                |
| CIDEr   | 0.84669               |

---

## Project Structure

```bash
├── dataset/
├── models/
├── preprocessing/
├── feature_extraction/
├── caption_generation/
├── evaluation/
├── outputs/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repository-link>
cd video-captioning-bilstm
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python train.py
```

### Generate Captions

```bash
python app.py
```

---

## Future Improvements

* Reinforcement learning-based optimization
* Adaptive beam search decoding
* Enhanced spatio-temporal grounding
* Multi-level paragraph summarization
* Large Vision Language Model integration

---

## Applications

* Video summarization
* Assistive technologies
* Surveillance analysis
* Educational video understanding
* Human-computer interaction

---


