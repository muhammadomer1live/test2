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
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox
import psutil

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
    caption = []
    for token in tokens:
        word = idx_to_word.get(token, '')
        if word in ['<PAD>', '<START>', '<END>', '<UNK>']:
            continue
        caption.append(word)
    return ' '.join(caption)

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
coco_train = COCO(f"{coco_root}/annotations/captions_train2017.json")

# Build vocabulary from COCO dataset
word_to_idx, idx_to_word = build_coco_vocab(coco_train)

# Create a subset of the dataset for quick testing
def create_subset(dataset, max_size=100):
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))
    return Subset(dataset, indices)

def load_train_dataset(max_size):
    train_dataset = datasets.CocoCaptions(
        root=f"{coco_root}/train2017",
        annFile=f"{coco_root}/annotations/captions_train2017.json",
        transform=transform
    )
    return create_subset(train_dataset, max_size)

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

# Define SCST Model
class SCSTModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(SCSTModel, self).__init__()
        self.cnn = cnn_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.critic = nn.Linear(hidden_dim, 1)  # Critic to evaluate the generated sequences

    def forward(self, images, captions):
        features = self.cnn(images).unsqueeze(1)
        
        captions_embed = self.embedding(captions)
        
        features = features.repeat(1, captions_embed.size(1), 1)
        
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)
        
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)
        return outputs

    def evaluate(self, images, captions):
        features = self.cnn(images).unsqueeze(1)
        captions_embed = self.embedding(captions)
        features = features.repeat(1, captions_embed.size(1), 1)
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)
        outputs, _ = self.rnn(inputs)
        return self.critic(outputs[:, -1, :])  # Critic score for the generated caption

# Using ResNet-18 (lighter model)
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()  # Removing the final fully connected layer

# Vocabulary size
vocab_size = len(word_to_idx)

# Initialize models
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
scst_model = SCSTModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

# Use CPU
device = 'cpu'
nic_model = nic_model.to(device)
scst_model = scst_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer_nic = optim.Adam(nic_model.parameters(), lr=0.001)
optimizer_scst = optim.Adam(scst_model.parameters(), lr=0.001)

# Save the model and vocab
def save_model_and_vocab(model, word_to_idx, idx_to_word, model_path='model.pth', vocab_path='vocab.pth'):
    torch.save(model.state_dict(), model_path)
    vocab = {'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word}
    torch.save(vocab, vocab_path)
    print(f"Model and vocabulary saved to {model_path} and {vocab_path}")

# Save models separately
def save_models(nic_model_path='nic_model.pth', scst_model_path='scst_model.pth'):
    save_model_and_vocab(nic_model, word_to_idx, idx_to_word, nic_model_path, 'nic_vocab.pth')
    save_model_and_vocab(scst_model, word_to_idx, idx_to_word, scst_model_path, 'scst_vocab.pth')

# Load the model and vocab
def load_model_and_vocab(model, model_path='model.pth', vocab_path='vocab.pth'):
    model.load_state_dict(torch.load(model_path))
    vocab = torch.load(vocab_path)
    return vocab['word_to_idx'], vocab['idx_to_word']

# Load models separately
def load_models(nic_model_path='nic_model.pth', scst_model_path='scst_model.pth'):
    nic_vocab = load_model_and_vocab(nic_model, nic_model_path, 'nic_vocab.pth')
    scst_vocab = load_model_and_vocab(scst_model, scst_model_path, 'scst_vocab.pth')
    return nic_vocab, scst_vocab

# Training loop with progress GUI and resource tracking
def train_model_gui(model_type='nic', num_epochs=2):
    global epoch_losses, batch_losses
    model = nic_model if model_type == 'nic' else scst_model
    optimizer = optimizer_nic if model_type == 'nic' else optimizer_scst

    # Create GUI window
    root = tk.Tk()
    root.title("Training Progress")

    # Add GUI elements for training parameters
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    # Epochs input
    tk.Label(input_frame, text="Number of Epochs:").grid(row=0, column=0, pady=5)
    epochs_entry = tk.Entry(input_frame)
    epochs_entry.grid(row=0, column=1, pady=5)

    # Max size input
    tk.Label(input_frame, text="Max Size in Dataset:").grid(row=1, column=0, pady=5)
    max_size_entry = tk.Entry(input_frame)
    max_size_entry.grid(row=1, column=1, pady=5)

    # Training mode selection
    mode_var = tk.StringVar(value='nic')
    tk.Radiobutton(input_frame, text="NIC Training", variable=mode_var, value='nic').grid(row=2, column=0, pady=5)
    tk.Radiobutton(input_frame, text="SCST Training", variable=mode_var, value='scst').grid(row=2, column=1, pady=5)

    def start_training():
        try:
            num_epochs = int(epochs_entry.get())
            max_size = int(max_size_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for epochs and max size.")
            return
        
        # Reload train dataset with the new max size
        train_dataset = load_train_dataset(max_size)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                                  collate_fn=lambda batch: coco_collate_fn(batch, word_to_idx, max_length=20))

        # Training progress
        progress_frame = ttk.Frame(root, padding="10")
        progress_frame.grid(row=3, column=0, padx=10, pady=10)
        
        epoch_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=400, mode='determinate')
        epoch_progress.grid(row=0, column=0, pady=5)
        epoch_label = tk.Label(progress_frame, text="Epoch Progress")
        epoch_label.grid(row=1, column=0, pady=5)

        batch_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=400, mode='determinate')
        batch_progress.grid(row=2, column=0, pady=5)
        batch_label = tk.Label(progress_frame, text="Batch Progress")
        batch_label.grid(row=3, column=0, pady=5)
        
        time_remaining_label = tk.Label(progress_frame, text="Estimated Time Remaining: Calculating...")
        time_remaining_label.grid(row=4, column=0, pady=5)

        def update_progress(epoch, batch, num_batches, start_time):
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / ((epoch * len(train_loader)) + batch + 1)) * (num_epochs * len(train_loader))
            remaining_time = estimated_total_time - elapsed_time

            epoch_progress['value'] = (epoch + 1) / num_epochs * 100
            batch_progress['value'] = (batch + 1) / len(train_loader) * 100
            time_remaining_label.config(text=f"Estimated Time Remaining: {int(remaining_time // 60)}m {int(remaining_time % 60)}s")
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

                    if model_type == 'nic':
                        loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].contiguous().view(-1))
                        loss.backward()
                        optimizer.step()
                    elif model_type == 'scst':
                        loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].contiguous().view(-1))
                        # Compute rewards and update model (simplified example)
                        rewards = torch.randn_like(loss)  # Dummy rewards
                        loss = -loss * rewards
                        loss.backward()
                        optimizer.step()

                    batch_losses.append(loss.item())
                    epoch_loss += loss.item()
                    num_batches += 1

                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

                    update_progress(epoch, batch_idx, len(train_loader), start_time)

            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')

        total_duration = time.time() - start_time
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
        print(f'Total Training Time: {total_duration:.2f} seconds')
        print(f'CPU Usage: {cpu_usage}%')
        print(f'Memory Usage: {memory_usage:.2f} MB')

        # Display summary in GUI
        summary_frame = ttk.Frame(root, padding="10")
        summary_frame.grid(row=5, column=0, padx=10, pady=10)

        time_label = tk.Label(summary_frame, text=f"Total Training Time: {total_duration:.2f} seconds")
        time_label.grid(row=0, column=0, pady=5)
        
        final_cpu_label = tk.Label(summary_frame, text=f"Final CPU Usage: {cpu_usage}%")
        final_cpu_label.grid(row=1, column=0, pady=5)
        
        final_memory_label = tk.Label(summary_frame, text=f"Final Memory Usage: {memory_usage:.2f} MB")
        final_memory_label.grid(row=2, column=0, pady=5)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker='o', linestyle='-', color='r')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Batch')
        plt.grid(True)
        plt.show()

        # Save the model and vocab
        save_models()

    start_button = tk.Button(input_frame, text="Start Training", command=lambda: start_training())
    start_button.grid(row=3, columnspan=2, pady=10)

    root.mainloop()

# Plot example captions after training
def plot_example_captions(model, dataset, word_to_idx, idx_to_word, num_examples=1, max_length=20):
    model.eval()  # Set the model to evaluation mode

    # Create subplots
    fig, axes = plt.subplots(num_examples, 1, figsize=(8, num_examples * 3))
    
    # Ensure axes is always iterable
    if num_examples == 1:
        axes = [axes]  # Convert single Axes object to a list

    for i in range(num_examples):
        image, caption = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        # Generate caption using the model
        with torch.no_grad():
            # Start with the <START> token
            caption_tokens = [word_to_idx['<START>']]
            for _ in range(max_length):
                input_tokens = torch.tensor([caption_tokens], device=device)
                outputs = model(image, input_tokens)
                predicted_id = outputs[:, -1, :].argmax(dim=-1).item()
                
                # Append predicted token to the caption
                caption_tokens.append(predicted_id)
                
                # Stop if <END> token is generated
                if predicted_id == word_to_idx['<END>']:
                    break
        
        # Convert tokens to words
        predicted_caption = detokenize_caption(caption_tokens, idx_to_word)

        # Display the image and both captions (predicted and original)
        axes[i].imshow(image.cpu().squeeze(0).permute(1, 2, 0))
        axes[i].set_title(f"Predicted: {predicted_caption}\nOriginal: {caption[0]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
train_model_gui(model_type='nic', num_epochs=2)

# After training, you can call plot_example_captions
# Load a dataset again to plot examples
train_dataset = load_train_dataset(max_size=5)  # Adjust max_size if needed
plot_example_captions(nic_model, train_dataset, word_to_idx, idx_to_word, num_examples=1)
