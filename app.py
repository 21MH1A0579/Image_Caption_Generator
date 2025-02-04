# from flask import Flask, render_template, request, redirect, url_for
# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle

# app = Flask(__name__)

# # Load the pre-trained model
# model = tf.keras.models.load_model('mymodel.h5')

# # Load the tokenizer
# with open('tokenizer.pkl', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)

# # Maximum caption length
# max_length = 35  # Update this based on your model

# # Function to get word from index
# def get_word_from_index(index, tokenizer):
#     return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

# # Function to predict caption
# def predict_caption(model, image_features, tokenizer, max_length):
#     caption = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([caption])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([image_features, sequence], verbose=0)
#         predicted_index = np.argmax(yhat)
#         predicted_word = get_word_from_index(predicted_index, tokenizer)
#         caption += " " + predicted_word
#         if predicted_word is None or predicted_word == 'endseq':
#             break
#     return caption

# # Function to extract image features using VGG16
# def extract_features(image_path):
#     vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     features = vgg_model.predict(image, verbose=0)
#     return features

# # Home route
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return redirect(request.url)
#         file = request.files['image']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             # Save the uploaded image
#             image_path = os.path.join('static', 'uploads', file.filename)
#             file.save(image_path)
            
#             # Extract image features
#             image_features = extract_features(image_path)
            
#             # Generate caption
#             caption = predict_caption(model, image_features, tokenizer, max_length)
            
#             # Render the result
#             return render_template('index.html', image_path=image_path, caption=caption)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load pre-trained components
model = tf.keras.models.load_model('mymodel.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load VGG16 for feature extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Define max_length from training
max_length = 37  # Adjust this based on training data

# Extract image features
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return vgg_model.predict(image, verbose=0)

# Generate caption
def generate_caption(image_features):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = {index: word for word, index in tokenizer.word_index.items()}.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', caption="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', caption="No selected file")
        
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        image_features = extract_features(filepath)
        caption = generate_caption(image_features)
        
        return render_template('index.html', image=filepath, caption=caption)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
