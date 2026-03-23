
<h1 align="center">Malay Sign Language Recognition</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Web%20App-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Holistic-0F9D58?style=for-the-badge" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/LSTM-Sequence%20Model-FF6F00?style=for-the-badge" />
</p>

<p align="center">
  A Computer Vision course project for recognizing dynamic Malay sign language gestures from video input.
</p>

## Project Introduction

This project is a **Malay Sign Language Recognition System** developed for a **Computer Vision course**. It aims to recognize dynamic Malay sign language gestures from video input by combining **computer vision**, **landmark extraction**, and **deep learning-based sequence modeling**.

The project focuses on sign language understanding from human motion. Instead of using raw image frames directly for classification, the system first extracts structured body and hand keypoints from each video frame, then models the temporal movement pattern of the gesture using an **LSTM-based neural network**. This design makes the project more interpretable and more suitable for gesture-based sequence recognition tasks.

In addition to model development, the project also includes a lightweight **FastAPI web application** that allows users to interact with the system through a browser. Users can upload a gesture video and receive the predicted Malay sign label with confidence information.

Overall, this repository demonstrates an end-to-end workflow for sign language recognition, including:
- video preprocessing,
- landmark extraction,
- sequence construction,
- deep learning classification,
- and simple web deployment.

---

## Project Goal

The goal of this project is to explore how computer vision can be applied to **sign language recognition**, especially for **Malay Sign Language**, and to build a working prototype that can classify gesture videos into meaningful sign categories.

This project also reflects the broader value of AI in:
- accessibility support,
- gesture-based human-computer interaction,
- and communication assistance technologies.

---

## Project Highlights

- Recognizes **Malay sign language gestures** from video input
- Uses **MediaPipe Holistic** to extract pose and hand landmarks
- Uses **LSTM** to model temporal motion patterns in gestures
- Includes preprocessing scripts for converting videos into keypoint sequences
- Provides a **FastAPI-based web interface** for user interaction
- Supports both **video upload prediction** and **sequence-based prediction**

---

## System Workflow

The overall workflow of the project can be divided into four main stages:

### 1. Video Input
The system takes a sign language video clip as input. Each video contains a dynamic gesture representing one Malay sign.

### 2. Landmark Extraction
For each frame, the system uses **MediaPipe Holistic** to detect:
- body pose landmarks,
- left hand landmarks,
- right hand landmarks.

These landmarks are converted into a fixed-length numerical feature vector.

### 3. Sequence Modeling
The extracted keypoints from multiple frames are combined into a sequence.  
This sequence is then fed into an **LSTM model**, which learns the temporal pattern of the gesture.

### 4. Gesture Prediction
The trained model predicts the corresponding Malay sign class.  
The result is returned together with a confidence score.

---

## Technical Overview

### Landmark Representation
The model input is based on:
- **33 pose landmarks** × 4 values `(x, y, z, visibility)`
- **21 left-hand landmarks** × 3 values `(x, y, z)`
- **21 right-hand landmarks** × 3 values `(x, y, z)`

This gives a total input size of **258 features per frame**.

### Temporal Input
The video is converted into a sequence of **30 frames**.  
If the valid gesture segment is longer than 30 frames, frames are uniformly sampled.  
If it is shorter than 30 frames, zero-padding is applied.

### Model
The classifier is built with a **2-layer LSTM** followed by a lightweight fully connected classification head.  
This architecture is suitable for capturing motion dynamics in gesture sequences.

---

## Technologies Used

- **Python**
- **FastAPI**
- **PyTorch**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- **Jinja2 / HTML**

---

## Project Structure

```bash
Malay-sign-language-recognition/
│
├── Train Sign Language Model.ipynb   # Notebook for training experiments
├── analyze.py                        # Analysis script
├── best_model.pth                    # Trained model weights
├── data_process.py                   # Video preprocessing and landmark extraction pipeline
├── main.py                           # FastAPI application entry point
├── model.py                          # LSTM model definition
├── test.py                           # Testing script
├── train.py                          # Model training script
├── utils.py                          # Utility functions for keypoint extraction and video processing
└── README.md                         # Project documentation
```

---

## File Description

### `main.py`
This is the deployment entry point of the project.

It creates a **FastAPI** application and provides:
- a homepage route,
- a guide page route,
- an about page route,
- a `/predict_stream` endpoint for sequence-based prediction,
- and an `/upload_video` endpoint for uploaded video prediction.

It also loads the trained model (`best_model.pth`) at application startup.

### `model.py`
This file defines the gesture classification model.

The model is based on:
- a **2-layer LSTM**
- followed by a small classifier with fully connected layers, ReLU, and dropout.

It is designed to classify gesture sequences into Malay sign categories.

### `utils.py`
This file contains the core utility functions used throughout the project, including:
- MediaPipe-based detection,
- landmark drawing,
- landmark extraction,
- video processing,
- and sequence normalization.

It also defines:
- the list of gesture classes,
- input size,
- hidden size,
- and total number of output classes.

### `data_process.py`
This script is used to preprocess the raw video dataset.

Its responsibilities include:
- reading gesture videos from dataset folders,
- extracting landmarks frame by frame,
- saving intermediate landmark files,
- and preparing training-ready sequence data.

### `train.py`
This script is intended for model training.  
It is used to train the LSTM classifier on the processed gesture sequence data.

### `test.py`
This script is used for testing or validating the trained recognition model.

### `analyze.py`
This file is used for additional analysis or result inspection.

### `best_model.pth`
This is the trained model checkpoint used during inference.

### `Train Sign Language Model.ipynb`
This notebook version is useful for experimentation, development, and step-by-step training demonstration.

---

## Web Application

The project includes a simple browser-based interface built with **FastAPI** and **Jinja2 templates**.

The web application supports:
- viewing the homepage,
- reading project guidance,
- reading project introduction,
- uploading a sign language video for prediction,
- receiving gesture prediction results with confidence.

This makes the project easier to demonstrate in a classroom or portfolio setting.

---

## Prediction Modes

### 1. Uploaded Video Prediction
A user uploads a gesture video file.  
The system:
- saves the file temporarily,
- extracts the keypoint sequence,
- runs the trained model,
- and returns the predicted gesture label.

### 2. Stream / Sequence Prediction
A sequence of extracted features can also be sent directly to the backend.  
The model then returns:
- the predicted gesture,
- and the confidence score.

---

## Why This Project Matters

This project is meaningful not only as a technical implementation, but also as an accessibility-related AI application.

It shows how computer vision can be used to:
- understand human gestures,
- support sign language communication,
- and build intelligent systems that interact with body motion.

As a student project, it is valuable because it connects:
- deep learning,
- sequence modeling,
- feature engineering,
- computer vision,
- and web deployment  
into one complete end-to-end system.

---

## Possible Future Improvements

This project can be further improved in several directions, such as:
- expanding the gesture vocabulary,
- improving dataset quality and class balance,
- adding real-time webcam-based inference,
- improving the frontend interface,
- and exploring more advanced sequence models such as GRU or Transformer-based architectures.

---

## Conclusion

This project presents an end-to-end implementation of **Malay Sign Language Recognition** using computer vision and deep learning.

By combining:
- **MediaPipe landmark extraction**,
- **LSTM-based temporal modeling**,
- and a **FastAPI web interface**,

the system provides a clear and practical demonstration of how AI can be used to recognize sign language gestures from video data.

It is a meaningful and well-structured Computer Vision course project with both technical depth and real-world application value.
