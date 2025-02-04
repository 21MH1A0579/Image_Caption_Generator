import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Function to extract image features
def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

# Load VGG16 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Define image directory
image_dir = 'Images'
image_features = {}

# Process each image
for img_name in tqdm(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, img_name)
    image_id = img_name.split('.')[0]
    image_features[image_id] = extract_features(img_path, vgg_model)

# Save extracted features
with open('features.pkl', 'wb') as f:
    pickle.dump(image_features, f)

print("Feature extraction complete. Saved as 'features.pkl'")
