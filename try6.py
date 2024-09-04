import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from pycocotools.coco import COCO
import numpy as np
import time
import os
from PIL import Image, ImageTk
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for progress bars
import tkinter as tk
from tkinter import ttk

# Set up for plotting
sns.set(style="whitegrid")

# Global lists to store loss values for plotting
epoch_losses = []
batch_losses = []

# Function to build vocabulary from COCO captions
def build_coco_vocab(coco, min_freq=5):
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    
    captions = [ann['caption'] for ann in anns]
    tokenized_captions = [caption.lower().split() for caption in captions]
    
    word_counter = Counter(word for caption in tokenized_captions for word in caption)
    vocab = [word for word, freq in word_counter.items() if freq >= min_freq]
    vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + vocab
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

# Tokenization function
def tokenize_caption(caption, word_to_idx, max_length=20):
    tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in caption.lower().split()]
    tokens = tokens[:max_length]
    tokens += [word_to_idx['<PAD>']] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

# Detokenization function excluding special tokens
def detokenize_caption(tokens, idx_to_word):
    return ' '.join(idx_to_word.get(token, '') for token in tokens if token not in [word_to_idx['<PAD>'], word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<UNK>']])

# Collate function for DataLoader
def coco_collate_fn(batch, word_to_idx, max_length=20):
    images = []
    captions = []
    
    for image, caption in batch:
        images.append(image)
        tokenized_caption = tokenize_caption(caption[0], word_to_idx, max_length)
        captions.append(tokenized_caption)
    
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    return images, captions

# Load COCO dataset with reduced image size
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define instance of COCO class to access the dataset
coco_train = COCO("{}/annotations/captions_train2017.json".format(coco_root))

# Build vocabulary from COCO dataset
word_to_idx, idx_to_word = build_coco_vocab(coco_train)

# Create a subset of the dataset for quick testing
def create_subset(dataset, max_size=100):
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))
    return Subset(dataset, indices)

# Note: datasets.CocoCaptions should be used to load the dataset
train_dataset = datasets.CocoCaptions(
    root="{}/train2017".format(coco_root),
    annFile="{}/annotations/captions_train2017.json".format(coco_root),
    transform=transform
)

train_dataset = create_subset(train_dataset, max_size=100)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                          collate_fn=lambda batch: coco_collate_fn(batch, word_to_idx, max_length=20))

# Define the image captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.cnn = cnn_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, images, captions):
        features = self.cnn(images).unsqueeze(1)
        
        captions_embed = self.embedding(captions)
        
        features = features.repeat(1, captions_embed.size(1), 1)
        
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)
        
        outputs, _ = self.rnn(inputs)
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

# Training loop with progress GUI
def train_model_gui(model, train_loader, optimizer, criterion, num_epochs=2):
    global epoch_losses, batch_losses
    model.train()
    
    root = tk.Tk()
    root.title("Training Progress")

    progress_frame = ttk.Frame(root, padding="10")
    progress_frame.grid(row=0, column=0, padx=10, pady=10)

    epoch_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=400, mode='determinate')
    epoch_progress.grid(row=0, column=0, pady=5)
    epoch_label = tk.Label(progress_frame, text="Epoch Progress")
    epoch_label.grid(row=1, column=0, pady=5)

    batch_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=400, mode='determinate')
    batch_progress.grid(row=2, column=0, pady=5)
    batch_label = tk.Label(progress_frame, text="Batch Progress")
    batch_label.grid(row=3, column=0, pady=5)

    def update_progress(epoch, batch, num_batches):
        epoch_progress['value'] = (epoch + 1) / num_epochs * 100
        batch_progress['value'] = (batch + 1) / num_batches * 100
        root.update_idletasks()

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (images, captions) in enumerate(train_loader):
                images = images.to(device)
                captions = captions.to(device)

                optimizer.zero_grad()
                outputs = model(images, captions[:, :-1])

                loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

                update_progress(epoch, batch_idx, len(train_loader))

        average_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(average_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_epoch_loss}')

        epoch_duration = time.time() - epoch_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_left = remaining_epochs * epoch_duration
        print(f'Epoch {epoch+1} completed in {epoch_duration:.2f} seconds. Estimated time left: {estimated_time_left:.2f} seconds.')

    total_duration = time.time() - start_time
    print(f'Training completed in {total_duration:.2f} seconds.')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses, marker='o', linestyle='-', color='r')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.grid(True)
    plt.show()

    root.destroy()

# Train the model with GUI
train_model_gui(model, train_loader, optim.Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss(), num_epochs=2)

# Save the model
model_save_path = '/mnt/c/Users/muham/Downloads/coco7/image_captioning_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Function to generate a caption for a given image
def generate_caption(model, image, word_to_idx, idx_to_word, max_length=30):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        caption = torch.tensor([word_to_idx['<START>']], dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            outputs = model(image, caption)
            next_word = outputs.argmax(dim=2)[:, -1].item()
            caption = torch.cat((caption, torch.tensor([[next_word]], dtype=torch.long).to(device)), dim=1)
            if next_word == word_to_idx['<END>']:
                break

        caption = caption.squeeze().cpu()
        return detokenize_caption(caption, idx_to_word)

# Function to plot example captions
def plot_example_captions(model, data_loader, idx_to_word):
    model.eval()
    with torch.no_grad():
        images, captions = next(iter(data_loader))
        images = images.to(device)
        captions = captions.to(device)

        fig, ax = plt.subplots(figsize=(6, 6))
        image = transforms.ToPILImage()(images[0].cpu())
        caption = detokenize_caption(captions[0].tolist(), idx_to_word)
        ax.imshow(image)
        ax.set_title(caption, fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        plt.show()

# Plot example caption
plot_example_captions(model, train_loader, idx_to_word)
