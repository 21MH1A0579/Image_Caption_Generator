# Image Captioning with VGG16 and LSTM

## Overview
This project focuses on building an AI-powered image captioning system that generates meaningful textual descriptions for images. By leveraging deep learning models, the system extracts features from input images and generates captions using a trained LSTM model.

## Key Features
- **Feature Extraction:** Uses the VGG16 model to extract a 4096-dimensional feature vector for each image.
- **Caption Generation:** Employs an LSTM-based model to generate meaningful captions from the extracted features.
- **Tokenizer Module:** Converts captions into sequences of integers for model training and reverses the process during inference.
- **Web Application:** A Flask-based web interface for uploading images and displaying generated captions.
- **Training Module:** Combines extracted image features with tokenized captions for training the LSTM model.

## System Architecture
The system is composed of the following key modules:

1. **Feature Extraction Module:** Utilizes VGG16 to extract feature vectors from input images.
2. **Caption Generation Module:** Generates captions using an LSTM model.
3. **Tokenizer Module:** Converts captions into sequences of integers and vice versa.
4. **Flask Web Application Module:** Provides a user-friendly interface for interacting with the model.
5. **Training Module:** Trains the LSTM model using image features and tokenized captions.

## Objective
The primary goal of this project is to develop a robust image captioning system that combines CNNs for image understanding and RNNs for text generation. This system automates image analysis, enabling better image dataset management and understanding.

## Requirements
- Python 3.8+
- TensorFlow/Keras
- Flask
- Numpy
- Pandas
- Jupyter Notebook
- Tokenizer library

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run the Flask web application:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
6. Upload an image to see the generated caption.

## Project Highlights
- **Deep Learning Models:** Combines CNN (VGG16) for feature extraction and RNN (LSTM) for caption generation.
- **End-to-End Workflow:** Includes feature extraction, model training, and a web application for real-time caption generation.
- **Real-Time Processing:** Generates captions dynamically for uploaded images.

## Contribution
Feel free to fork this repository and submit pull requests. Contributions are welcome for:
- Improving caption quality
- Optimizing model performance
- Enhancing the web application interface

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

