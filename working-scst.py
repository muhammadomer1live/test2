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
import seaborn as sns

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

coco_train = COCO("{}/annotations/captions_train2017.json".format(coco_root))

word_to_idx, idx_to_word = build_coco_vocab(coco_train)

def create_subset(dataset, max_size=100):
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))
    return Subset(dataset, indices)

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
        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_input = nn.Linear(512 + embedding_dim, 256)
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

cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()

vocab_size = len(word_to_idx)
model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
device = 'cpu'
model = model.to(device)

# Greedy search for baseline captions
def greedy_search(model, image, max_length=20):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        caption = torch.tensor([word_to_idx['<START>']], dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            outputs = model(image, caption)
            next_word = outputs.argmax(dim=2)[:, -1].item()
            caption = torch.cat((caption, torch.tensor([[next_word]], dtype=torch.long).to(device)), dim=1)
            if next_word == word_to_idx['<END>']:
                break

        return caption.squeeze().cpu()

# Random sampling of captions
def sample_caption(model, image, max_length=20):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        caption = torch.tensor([word_to_idx['<START>']], dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            outputs = model(image, caption)
            prob_distribution = torch.nn.functional.softmax(outputs[:, -1, :], dim=-1)
            next_word = torch.multinomial(prob_distribution, 1).item()
            caption = torch.cat((caption, torch.tensor([[next_word]], dtype=torch.long).to(device)), dim=1)
            if next_word == word_to_idx['<END>']:
                break

        return caption.squeeze().cpu()

# SCST training loop
def train_model_scst(model, train_loader, optimizer, num_epochs=2, reward_func=None):
    global epoch_losses, batch_losses
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for images, captions in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            
            # Get baseline caption using greedy search
            with torch.no_grad():
                baseline_captions = [greedy_search(model, img) for img in images]

            # Get sampled caption using random sampling
            sampled_captions = [sample_caption(model, img) for img in images]

            # Calculate reward (e.g., CIDEr, BLEU)
            rewards = torch.tensor([reward_func(sampled, baseline) for sampled, baseline in zip(sampled_captions, baseline_captions)])

            # Compute log probabilities for sampled captions
            outputs = model(images, captions[:, :-1])
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            log_probs = log_probs.gather(2, captions[:, 1:].unsqueeze(-1)).squeeze(-1)

            # SCST loss: -(reward - baseline reward) * log_probs
            scst_loss = -(rewards - rewards.mean()) * log_probs.sum(dim=1)
            loss = scst_loss.mean()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}')

        # Plotting the training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), epoch_losses, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.show()

        # Plot loss per batch
        plt.figure(figsize=(10, 5))
        plt.plot(batch_losses, marker='o', linestyle='-', color='r')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Batch')
        plt.grid(True)
        plt.show()

# Sample reward function using CIDEr (placeholders)
def reward_func(sampled_caption, baseline_caption):
    return np.random.rand()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model using SCST
train_model_scst(model, train_loader, optimizer, num_epochs=2, reward_func=reward_func)

# Save the model
model_save_path = '/mnt/c/Users/muham/Downloads/coco7/image_captioning_model_scst.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Function to generate a caption for a given image
def generate_caption(model, image, word_to_idx, idx_to_word, max_length=20):
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

# Test with a random image
def test_random_image(model, dataset, word_to_idx, idx_to_word):
    model.eval()
    with torch.no_grad():
        random_idx = np.random.randint(0, len(dataset))
        image, _ = dataset[random_idx]
        image = transform(image).unsqueeze(0).to(device)
        
        generated_caption = generate_caption(model, image, word_to_idx, idx_to_word)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        image_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
        ax.imshow(image_pil)
        ax.set_title(f'Generated Caption: {generated_caption}', fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        plt.show()

# Test a random image from the dataset
test_random_image(model, train_dataset, word_to_idx, idx_to_word)
