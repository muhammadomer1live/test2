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

# Global lists to store loss values for plotting
epoch_losses = []
val_losses = []
train_losses = epoch_losses

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

def create_subset(dataset, max_size=1000):
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

train_dataset = load_dataset(f"{coco_root}/train2017", f"{coco_root}/annotations/captions_train2017.json", max_size=1000)
val_dataset = load_dataset(f"{coco_root}/val2017", f"{coco_root}/annotations/captions_val2017.json", max_size=1000)

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

cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()

vocab_size = len(word_to_idx)
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
scst_model = SCSTModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

device = 'cpu'
nic_model = nic_model.to(device)
scst_model = scst_model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer_nic = optim.Adam(nic_model.parameters(), lr=0.001)
optimizer_scst = optim.Adam(scst_model.parameters(), lr=0.001)

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

# Global lists to store loss values for plotting

def train_and_validate_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10, patience=3):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

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
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

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

    # Save global lists
    epoch_losses.append(train_losses)
    val_losses.append(val_losses)

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

# Train NIC model
# train_and_validate_model(nic_model, train_dataloader, val_dataloader, criterion, optimizer_nic, num_epochs=10, patience=3)

# Load best model and evaluate
nic_model.load_state_dict(torch.load('best_model.pth'))
#rouge_scores, bleu_score = evaluate_model_with_rouge_and_bleu(nic_model, val_dataloader, word_to_idx, idx_to_word)
#print(f"ROUGE Scores: {rouge_scores}")
#print(f"BLEU Score: {bleu_score:.4f}")

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
