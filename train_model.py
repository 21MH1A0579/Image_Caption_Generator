import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
import tensorflow as tf

# Load captions from file
def load_captions(captions_file):
    with open(captions_file, 'r') as file:
        captions_doc = file.read()
    image_to_captions_mapping = defaultdict(list)
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        image_to_captions_mapping[image_id].append(caption)
    return image_to_captions_mapping

# Clean captions
def clean_captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = ''.join(char for char in caption if char.isalpha() or char.isspace())
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption

# Data generator
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                if n == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield [X1, X2], y
                    X1, X2, y = [], [], []
                    n = 0

# Load data and train model
def train_model():
    # Load captions
    captions_file = 'captions.txt'
    image_to_captions_mapping = load_captions(captions_file)
    clean_captions(image_to_captions_mapping)

    # Load pre-extracted features
    with open('features.pkl', 'rb') as f:
        image_features = pickle.load(f)

    # Tokenize captions
    all_captions = [caption for captions in image_to_captions_mapping.values() for caption in captions]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)

    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    # Split data
    image_ids = list(image_to_captions_mapping.keys())
    split = int(len(image_ids) * 0.9)
    train_keys, test_keys = image_ids[:split], image_ids[split:]

    # Define model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train model
    epochs = 10
    batch_size = 32
    steps = len(train_keys) // batch_size
    train_gen = data_generator(train_keys, image_to_captions_mapping, image_features, tokenizer, max_length, vocab_size, batch_size)
    
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps, verbose=1)

    # Save model
    model.save('mymodel.h5')

if __name__ == '__main__':
    train_model()
