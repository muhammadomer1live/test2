import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from pycocotools.coco import COCO
import numpy as np
import time
import os
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Tokenization function
def tokenize_caption(caption, word_to_idx, max_length=20):
    tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in caption.lower().split()]
    tokens = tokens[:max_length]  # Truncate if too long
    tokens += [word_to_idx['<PAD>']] * (max_length - len(tokens))  # Pad if too short
    return torch.tensor(tokens, dtype=torch.long)

# Detokenization function
def detokenize_caption(tokens, idx_to_word):
    return ' '.join(idx_to_word.get(token, '<UNK>') for token in tokens)

# Collate function for DataLoader
def coco_collate_fn(batch, word_to_idx, max_length=20):
    images = []
    captions = []
    
    for image, caption in batch:
        images.append(image)
        tokenized_caption = tokenize_caption(caption[0], word_to_idx, max_length)  # Tokenize the first caption
        captions.append(tokenized_caption)
    
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    captions = torch.stack(captions, dim=0)  # Stack captions into a single tensor
    return images, captions

# Load COCO dataset with reduced image size
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced image size
    transforms.ToTensor(),
])

# Define instance of COCO class to access the dataset
coco_train = COCO("{}/annotations/captions_train2017.json".format(coco_root))

# Simple vocabulary (update based on actual dataset)
word_to_idx = {
    '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
    'a': 4, 'cat': 5, 'dog': 6, 'on': 7, 'the': 8, 'mat': 9
}
idx_to_word = {v: k for k, v in word_to_idx.items()}  # Inverse dictionary for detokenization

# Create a subset of the dataset for quick testing
def create_subset(dataset, max_size=100):
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))  # Convert to int for compatibility
    return Subset(dataset, indices)

train_dataset = datasets.CocoCaptions(
    root="{}/train2017".format(coco_root),
    annFile="{}/annotations/captions_train2017.json".format(coco_root),
    transform=transform
)
train_dataset = create_subset(train_dataset, max_size=100)  # Limiting to 1000 images

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                          collate_fn=lambda batch: coco_collate_fn(batch, word_to_idx, max_length=20))  # Smaller batch size

# Define the image captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.cnn = cnn_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # New linear layer to project concatenated features to the correct input size
        self.fc_input = nn.Linear(512 + embedding_dim, 256)  # Image features + word embedding

        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, images, captions):
        # Extract image features using CNN
        features = self.cnn(images).unsqueeze(1)  # Shape: (batch_size, 1, 512)
        
        # Embed the captions
        captions_embed = self.embedding(captions)  # Shape: (batch_size, caption_length, embedding_dim)
        
        # Repeat image features along the caption length dimension
        features = features.repeat(1, captions_embed.size(1), 1)
        
        # Concatenate image features and caption embeddings
        inputs = torch.cat((features, captions_embed), dim=2)
        
        # Project concatenated features to the correct input size for LSTM
        inputs = self.fc_input(inputs)  # Shape: (batch_size, caption_length, 256)
        
        # Pass the inputs through the LSTM
        outputs, _ = self.rnn(inputs)
        
        # Project LSTM outputs to the vocabulary size
        outputs = self.fc(outputs)
        return outputs

# Using ResNet-18 (lighter model)
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()  # Removing the final fully connected layer

# Vocabulary size
vocab_size = len(word_to_idx)
model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

# Use CPU
device = 'cpu'
model = model.to(device)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, optimizer, criterion, num_epochs=2):
    model.train()
    start_time = time.time()  # Start time for the whole training process
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time for the current epoch
        for images, captions in train_loader:
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])  # Input to the model excludes the last token

            loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].contiguous().view(-1))  # Target is shifted by one
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

        epoch_duration = time.time() - epoch_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_left = remaining_epochs * epoch_duration
        print(f'Epoch {epoch+1} completed in {epoch_duration:.2f} seconds. Estimated time left: {estimated_time_left:.2f} seconds.')

    total_duration = time.time() - start_time
    print(f'Training completed in {total_duration:.2f} seconds.')

# Train the model (no validation, only train set)
train_model(model, train_loader, optimizer, criterion, num_epochs=2)

# Save the model
model_save_path = '/mnt/c/Users/muham/Downloads/coco7/image_captioning_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Function to generate a caption for a given image
def generate_caption(model, image, word_to_idx, idx_to_word, max_length=20):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        caption = torch.tensor([word_to_idx['<START>']], dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            outputs = model(image, caption)
            next_word = outputs.argmax(dim=2)[:, -1].item()  # Get the predicted word
            caption = torch.cat((caption, torch.tensor([[next_word]], dtype=torch.long).to(device)), dim=1)
            if next_word == word_to_idx['<END>']:
                break

        caption = caption.squeeze().cpu()
        return detokenize_caption(caption, idx_to_word)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Evaluation function
def evaluate_model(model, data_loader, coco):
    model.eval()
    all_references = {}
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(data_loader):
            images = images.to(device)

            for i in range(images.size(0)):
                # Obtain the COCO image ID directly from the dataset
                img_id = data_loader.dataset.dataset.ids[data_loader.dataset.indices[batch_idx * data_loader.batch_size + i]]

                # Convert the caption tokens to a string
                ref_caption = detokenize_caption(captions[i].tolist(), idx_to_word)
                if img_id not in all_references:
                    all_references[img_id] = []
                all_references[img_id].append(ref_caption)

                # Generate the caption
                output_caption = generate_caption(model, transforms.ToPILImage()(images[i]), word_to_idx, idx_to_word)

                # Append the result using the correct image ID
                all_results.append({
                    'image_id': img_id,
                    'caption': output_caption
                })

    # Convert results to COCO format
    coco_res = coco.loadRes(all_results)

    # Initialize COCO evaluation metrics
    bleu = Bleu(4)
    meteor = Meteor()
    cider = Cider()
    spice = Spice()

    # Compute scores
    bleu_score, _ = bleu.compute_score(all_references, all_results)
    meteor_score, _ = meteor.compute_score(all_references, all_results)
    cider_score, _ = cider.compute_score(all_references, all_results)
    spice_score, _ = spice.compute_score(all_references, all_results)

    # Display scores
    print(f'BLEU Score: {bleu_score}')
    print(f'METEOR Score: {meteor_score}')
    print(f'CIDEr Score: {cider_score}')
    print(f'SPICE Score: {spice_score}')

# Initialize COCO instance
coco = COCO("{}/annotations/captions_train2017.json".format(coco_root))

# Evaluate the model on the training set
evaluate_model(model, train_loader, coco)
