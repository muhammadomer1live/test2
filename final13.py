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
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import psutil
from rouge_score import rouge_scorer
from PIL import ImageTk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sns.set(style="whitegrid")

# Global lists to store various values
epoch_losses = []
val_losses = []
train_losses = epoch_losses

nic_rouge_scores = []
scst_rouge_scores = []
nic_bleu_score = []
scst_bleu_score = []
nic_caption_lengths = []
scst_caption_lengths = []
captions_nic = []
captions_scst = []

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

# Load COCO dataset with enhanced preprocessing
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load train and validation datasets
coco_train = COCO(f"{coco_root}/annotations/captions_train2017.json")
coco_val = COCO(f"{coco_root}/annotations/captions_val2017.json")
word_to_idx, idx_to_word = build_coco_vocab(coco_train)

def create_subset(dataset, max_size):
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))
    return Subset(dataset, indices)


def load_dataset(root, annFile, max_size):
    dataset = datasets.CocoCaptions(
        root=root,
        annFile=annFile,
        transform=transform
    )
    return create_subset(dataset, max_size)

train_dataset = load_dataset(f"{coco_root}/train2017", f"{coco_root}/annotations/captions_train2017.json", max_size=100)
val_dataset = load_dataset(f"{coco_root}/val2017", f"{coco_root}/annotations/captions_val2017.json", max_size=100)

def get_dataloader(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: coco_collate_fn(x, word_to_idx))

train_dataloader = get_dataloader(train_dataset)
val_dataloader = get_dataloader(val_dataset)

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

class SCSTModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(SCSTModel, self).__init__()
        self.cnn = cnn_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.critic = nn.Linear(hidden_dim, 1)

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
        return self.critic(outputs[:, -1, :])

class SATModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(SATModel, self).__init__()
        self.cnn = cnn_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_input = nn.Linear(hidden_dim + embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.attention_fc = nn.Linear(512, embedding_dim)  # Example Linear layer to match dimensions

    def forward(self, images, captions):
        features = self.cnn(images)  # Shape: (batch_size, 512)
        features = features.unsqueeze(1)  # Add a sequence dimension: (batch_size, 1, 512)

        captions_embed = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)

        # Apply attention mechanism
        attended_features = self.attention_fc(features)  # Shape: (batch_size, 1, embedding_dim)
        attended_features = attended_features.repeat(1, captions_embed.size(1), 1)  # Repeat along sequence dimension: (batch_size, seq_length, embedding_dim)

        inputs = torch.cat((attended_features, captions_embed), dim=2)  # Concatenate along feature dimension
        inputs = self.fc_input(inputs)

        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)
        return outputs

cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()

vocab_size = len(word_to_idx)
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
scst_model = SCSTModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
sat_model = SATModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

device = 'cpu'
nic_model = nic_model.to(device)
scst_model = scst_model.to(device)
sat_model = sat_model.to(device)


criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer_nic = optim.Adam(nic_model.parameters(), lr=0.001)
optimizer_scst = optim.Adam(scst_model.parameters(), lr=0.001)
optimizer_sat = optim.Adam(sat_model.parameters(), lr=0.001)

def compute_rouge_scores(hypothesis, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    num_references = len(references)
    
    for ref in references:
        score = scorer.score(ref, hypothesis)
        for key in scores:
            scores[key] += score[key].fmeasure
    
    for key in scores:
        scores[key] /= num_references
    
    return scores

def compute_bleu_scores(hypotheses, references):
    smooth = SmoothingFunction().method4
    bleu_scores = []
    
    for hyp, refs in zip(hypotheses, references):
        refs = [ref.split() for ref in refs]  # BLEU expects a list of lists of references
        hyp = hyp.split()
        score = sentence_bleu(refs, hyp, smoothing_function=smooth)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

def evaluate_model_with_rouge_and_bleu(model, dataloader, word_to_idx, idx_to_word):
    model.eval()
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)
            
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.cpu().numpy()
            captions = captions.cpu().numpy()
            
            for i in range(predicted.shape[0]):
                hyp = detokenize_caption(predicted[i], idx_to_word)
                ref = detokenize_caption(captions[i], idx_to_word)
                all_hypotheses.append(hyp)
                all_references.append(ref)
    
    rouge_scores = compute_rouge_scores(' '.join(all_hypotheses), all_references)
    bleu_score = compute_bleu_scores(all_hypotheses, all_references)
    return rouge_scores, bleu_score

def train_and_validate_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, patience=3):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    total_steps = num_epochs * len(train_dataloader)
    step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, captions) in enumerate(train_dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # Update progress bar
            step += 1
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / step) * (total_steps - step)
            print(f"\rEpoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {epoch_loss / (batch_idx + 1):.4f}, Remaining Time: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s", end='')

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        val_loss = evaluate_model_on_val_set(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses

def evaluate_model_on_val_set(model, val_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in val_dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

def evaluate_model_with_attention(model, dataloader, word_to_idx, idx_to_word):
    model.eval()
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)
            
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.cpu().numpy()
            captions = captions.cpu().numpy()
            
            for i in range(predicted.shape[0]):
                hyp = detokenize_caption(predicted[i], idx_to_word)
                ref = detokenize_caption(captions[i], idx_to_word)
                all_hypotheses.append(hyp)
                all_references.append(ref)
    
    rouge_scores = compute_rouge_scores(' '.join(all_hypotheses), all_references)
    bleu_score = compute_bleu_scores(all_hypotheses, all_references)
    return rouge_scores, bleu_score


def evaluate_and_save_metrics(model, dataloader, word_to_idx, idx_to_word, model_type):
    rouge_scores, bleu_score = evaluate_model_with_rouge_and_bleu(model, dataloader, word_to_idx, idx_to_word)
    if model_type == 'nic':
        nic_rouge_scores.append(rouge_scores)
        nic_bleu_score.append(bleu_score)
    elif model_type == 'scst':
        scst_rouge_scores.append(rouge_scores)
        scst_bleu_score.append(bleu_score)
    
    all_lengths = []
    all_captions = []

    with torch.no_grad():
        for images, captions in dataloader:
            captions = captions.cpu().numpy()
            for i in range(captions.shape[0]):
                caption_length = np.sum(captions[i] != word_to_idx['<PAD>'])
                all_lengths.append(caption_length)
                all_captions.append(' '.join([idx_to_word[idx] for idx in captions[i] if idx != word_to_idx['<PAD>']]))
    
    if model_type == 'nic':
        nic_caption_lengths.append(all_lengths)
        captions_nic.extend(all_captions)
    elif model_type == 'scst':
        scst_caption_lengths.append(all_lengths)
        captions_scst.extend(all_captions)

# Train NIC model
#train_and_validate_model(nic_model, train_dataloader, val_dataloader, criterion, optimizer_nic, num_epochs=5, patience=3)

# Evaluate NIC model
#nic_model.load_state_dict(torch.load('best_model.pth'))
#evaluate_and_save_metrics(nic_model, val_dataloader, word_to_idx, idx_to_word, model_type='nic')

# Train SCST model
#train_and_validate_model(scst_model, train_dataloader, val_dataloader, criterion, optimizer_scst, num_epochs=5, patience=3)

# Evaluate SCST model
#scst_model.load_state_dict(torch.load('best_model.pth'))
#evaluate_and_save_metrics(scst_model, val_dataloader, word_to_idx, idx_to_word, model_type='scst')

# Train SAT model
#train_and_validate_model(sat_model, train_dataloader, val_dataloader, criterion, optimizer_sat, num_epochs=5, patience=3)

#sat_model.load_state_dict(torch.load('best_model.pth'))
#evaluate_and_save_metrics(sat_model, val_dataloader, word_to_idx, idx_to_word, model_type='sat')

#rouge_scores, bleu_score = evaluate_model_with_attention(sat_model, val_dataloader, word_to_idx, idx_to_word)

#print(f"ROUGE Scores: {rouge_scores}")
#print(f"BLEU Score: {bleu_score}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Function to display an image with captions
def display_image_and_captions(image, original_caption, predicted_caption):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.title(f"Predicted Caption: {predicted_caption}")
    plt.axis('off')
    plt.show()

# Function to untransform the image
def untransform_image(image_tensor, transform):
    # Unnormalize
    unnormalize_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    ])
    
    image_tensor = unnormalize_transform(image_tensor)

    # Convert the tensor to a PIL image
    image_tensor = torch.clamp(image_tensor, 0, 1)  # Clamp values to [0, 1]
    return transforms.ToPILImage()(image_tensor)

# Function to detokenize a caption
def detokenize_caption(tokens, idx_to_word):
    return ' '.join([idx_to_word[idx] for idx in tokens if idx != 0])

def evaluate_model_and_show_predictions(model, dataloader, idx_to_word, device, transform):
    model.eval()
    
    with torch.no_grad():
        # Get a batch from the dataloader
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            # Generate prediction for the first image in the batch
            outputs = model(images, captions)  # Pass captions if required by the model
            _, predicted = torch.max(outputs, dim=2)
            predicted_caption = predicted[0].cpu().numpy()
            
            # Convert to human-readable format
            original_caption_text = detokenize_caption(captions[0].cpu().numpy(), idx_to_word)
            predicted_caption_text = detokenize_caption(predicted_caption, idx_to_word)
            
            # Convert the image tensor to PIL image
            image = untransform_image(images[0].cpu(), transform)
            
            # Display the image and captions
            display_image_and_captions(image, original_caption_text, predicted_caption_text)
            break  # Only show one image for this example

# Assuming you have already set up your model, dataloader, and device
# Example usage:
evaluate_model_and_show_predictions(nic_model, val_dataloader, idx_to_word, device, transform)

import json
import numpy as np

def convert_to_serializable(data):
    """
    Recursively converts NumPy data types to native Python data types for JSON serialization.
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy arrays to lists
    elif isinstance(data, np.generic):
        return data.item()  # Convert NumPy scalars to native Python scalars
    else:
        return data

def save_metrics_to_json(filename):
    # Prepare the data to be saved
    data = {
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'nic_rouge_scores': nic_rouge_scores,
        'scst_rouge_scores': scst_rouge_scores,
        'nic_bleu_score': nic_bleu_score,
        'scst_bleu_score': scst_bleu_score,
        'nic_caption_lengths': nic_caption_lengths,
        'scst_caption_lengths': scst_caption_lengths,
        'captions_nic': captions_nic,
        'captions_scst': captions_scst
    }
    
    # Convert data to serializable format
    serializable_data = convert_to_serializable(data)
    
    # Write the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=4)

# Example usage
save_metrics_to_json('metrics1.json')

def setup_gui():
    def on_start():
        max_size = int(max_size_entry.get())
        num_epochs = int(num_epochs_entry.get())
        
        # Start training with user inputs
        start_training(max_size, num_epochs)

    root = tk.Tk()
    root.title("Training Configuration")

    tk.Label(root, text="Data Size:").grid(row=0, column=0, padx=10, pady=10)
    max_size_entry = tk.Entry(root)
    max_size_entry.grid(row=0, column=1, padx=10, pady=10)
    
    tk.Label(root, text="No. of Epochs:").grid(row=1, column=0, padx=10, pady=10)
    num_epochs_entry = tk.Entry(root)
    num_epochs_entry.grid(row=1, column=1, padx=10, pady=10)

    start_button = tk.Button(root, text="Start Training", command=on_start)
    start_button.grid(row=2, columnspan=2, padx=10, pady=10)

    root.mainloop()

def start_training(max_size, num_epochs):
    train_dataset = load_dataset(f"{coco_root}/train2017", f"{coco_root}/annotations/captions_train2017.json", max_size)
    val_dataset = load_dataset(f"{coco_root}/val2017", f"{coco_root}/annotations/captions_val2017.json", max_size)

    train_dataloader = get_dataloader(train_dataset)
    val_dataloader = get_dataloader(val_dataset)

    # Train NIC model
    #train_and_validate_model(nic_model, train_dataloader, val_dataloader, criterion, optimizer_nic, num_epochs=num_epochs, patience=3)

    # Evaluate NIC model
    #nic_model.load_state_dict(torch.load('best_model.pth'))
    #evaluate_and_save_metrics(nic_model, val_dataloader, word_to_idx, idx_to_word, model_type='nic')

    # Train SCST model
    #train_and_validate_model(scst_model, train_dataloader, val_dataloader, criterion, optimizer_scst, num_epochs=num_epochs, patience=3)

    # Evaluate SCST model
    #scst_model.load_state_dict(torch.load('best_model.pth'))
    #evaluate_and_save_metrics(scst_model, val_dataloader, word_to_idx, idx_to_word, model_type='scst')

    # Train SAT model
    train_and_validate_model(sat_model, train_dataloader, val_dataloader, criterion, optimizer_sat, num_epochs=num_epochs, patience=3)

    sat_model.load_state_dict(torch.load('best_model.pth'))
    evaluate_and_save_metrics(sat_model, val_dataloader, word_to_idx, idx_to_word, model_type='sat')

    save_metrics_to_json('metrics1.json')

# Call the GUI setup function
setup_gui()
