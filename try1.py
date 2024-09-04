import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load captions file
annotation_file = '/mnt/c/Users/muham/Downloads/coco7/annotations/captions_train2017.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Create a mapping between image_id and captions
captions_map = {}
for annot in annotations['annotations']:
    image_id = annot['image_id']
    caption = annot['caption']
    if image_id not in captions_map:
        captions_map[image_id] = []
    captions_map[image_id].append(caption)

# Preprocess Images using InceptionV3
inception_model = InceptionV3(include_top=False, weights='imagenet')
image_model = Model(inputs=inception_model.input, outputs=inception_model.layers[-1].output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def extract_image_features(img_path):
    img = preprocess_image(img_path)
    features = image_model.predict(img)
    features = np.squeeze(features)
    return features

# Prepare a list of all captions
all_captions = []
for caps in captions_map.values():
    all_captions.extend(caps)

# Tokenize the captions
tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

# Convert captions to sequences
max_len = 35
def encode_caption(caption):
    seq = tokenizer.texts_to_sequences([caption])[0]
    return pad_sequences([seq], maxlen=max_len, padding='post')[0]

# Encoder model: CNN feature extractor
image_input = tf.keras.Input(shape=(8, 8, 2048))
image_features = tf.keras.layers.GlobalAveragePooling2D()(image_input)
image_features = Dense(256, activation='relu')(image_features)

# Decoder model: RNN for caption generation
caption_input = tf.keras.Input(shape=(max_len,))
caption_embedding = Embedding(input_dim=10000, output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)
decoder_output = Add()([image_features, caption_lstm])
output = Dense(10000, activation='softmax')(decoder_output)

# Combine Encoder and Decoder
model = Model(inputs=[image_input, caption_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_ids, captions_map, tokenizer, batch_size=64, max_len=35, image_dir='/mnt/c/Users/muham/Downloads/coco7/train2017'):
        self.image_ids = image_ids
        self.captions_map = captions_map
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.image_dir = image_dir
        self.indexes = np.arange(len(self.image_ids))
        np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_features = []
        batch_caption_sequences = []
        batch_next_words = []
        
        for i in batch_indexes:
            img_id = self.image_ids[i]
            img_path = os.path.join(self.image_dir, f"{img_id:012}.jpg")
            features = extract_image_features(img_path)
            features = features.flatten().reshape(1, -1)
            
            for caption in self.captions_map[img_id]:
                encoded_caption = encode_caption(caption)
                for j in range(1, len(encoded_caption)):
                    caption_seq = encoded_caption[:j]
                    next_word = encoded_caption[j]
                    
                    batch_image_features.append(features)
                    batch_caption_sequences.append(pad_sequences([caption_seq], maxlen=self.max_len, padding='post')[0])
                    batch_next_words.append(next_word)
        
        batch_image_features = np.vstack(batch_image_features)
        batch_caption_sequences = np.array(batch_caption_sequences)
        batch_next_words = np.array(batch_next_words)
        
        return (batch_image_features, batch_caption_sequences), tf.keras.utils.to_categorical(batch_next_words, num_classes=10000)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# Split image_ids into training and validation sets
train_image_ids, val_image_ids = train_test_split(list(captions_map.keys()), test_size=0.2, random_state=42)

# Instantiate the data generators
train_generator = DataGenerator(train_image_ids, captions_map, tokenizer, batch_size=64, max_len=max_len, image_dir='/mnt/c/Users/muham/Downloads/coco7/train2017')
val_generator = DataGenerator(val_image_ids, captions_map, tokenizer, batch_size=64, max_len=max_len, image_dir='/mnt/c/Users/muham/Downloads/coco7/train2017')

# Train the model using the training and validation generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,  # Adjust epochs as needed
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
