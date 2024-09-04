import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import json
from pycocotools.coco import COCO
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

# Set paths to your local COCO dataset
train_images = "/mnt/c/Users/muham/Downloads/coco7/train2017/"
val_images = "/mnt/c/Users/muham/Downloads/coco7/val2017/"
train_annotations = "/mnt/c/Users/muham/Downloads/coco7/annotations/captions_train2017.json"
val_annotations = "/mnt/c/Users/muham/Downloads/coco7/annotations/captions_val2017.json"

# Load COCO annotations
coco_train = COCO(train_annotations)
coco_val = COCO(val_annotations)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom COCO dataset class
class CocoDataset(Dataset):
    def __init__(self, root_dir, coco, transform=None):
        self.root_dir = root_dir
        self.coco = coco
        self.transform = transform
        self.ids = list(coco.imgs.keys())[:1000]  # Use a subset to reduce training time

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]

        return img, captions

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack([img for img in imgs])
    
    # Flatten captions and create a mapping from index to caption
    captions_flat = [caption for sublist in captions for caption in sublist]
    
    return imgs, captions_flat

# Load training and validation datasets
train_dataset = CocoDataset(train_images, coco_train, transform=transform)
val_dataset = CocoDataset(val_images, coco_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Load pre-trained ResNet18 and remove the classification layer
resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Removing last fully connected layer
resnet.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.add_word('<pad>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

    def get_word_index(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def get_index_word(self, idx):
        return self.idx2word.get(idx, '<unk>')

# Create vocabulary
vocab = Vocabulary()
# Example code to add words to the vocabulary
for _, captions in train_loader:
    for caption in captions:
        for word in caption.split():
            vocab.add_word(word)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Image captioning model with LSTM decoder
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = nn.Linear(512, embed_size)  # Linear projection of image features (ResNet18 output size is 512)
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer for captions
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Final linear layer to predict next word

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # Embed captions
        features = features.unsqueeze(1)  # Add batch dimension for image features
        # Ensure concatenation of features and embeddings matches dimensions
        features_expanded = features.expand(-1, embeddings.size(1), -1)
        inputs = torch.cat((features_expanded, embeddings), dim=2)  # Concatenate along feature dimension
        hiddens, _ = self.lstm(inputs)  # Pass through LSTM
        outputs = self.fc(hiddens)  # Final output to predict next word
        return outputs

    def sample(self, features, max_len=20, greedy=False):
        # Implement a sampling method to generate captions
        pass

# Define model hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1

# Initialize the model
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Pretrain the model with Cross-Entropy loss
def pretrain_cross_entropy(model, train_loader, resnet, device, optimizer, criterion, vocab_size, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, captions in train_loader:
            imgs = imgs.to(device)
            
            # Convert captions to indices
            captions_idx = [[vocab.get_word_index(word) for word in caption.split()] for caption in captions]
            max_len = max(len(caption) for caption in captions_idx)
            captions_idx = [caption + [vocab.get_word_index('<pad>')] * (max_len - len(caption)) for caption in captions_idx]
            captions_idx = torch.tensor(captions_idx).to(device)

            # Extract features using ResNet
            with torch.no_grad():
                features = resnet(imgs).squeeze()  # Should have shape [batch_size, 512]

            # Forward pass through the model
            outputs = model(features, captions_idx[:, :-1])  # Shape: [batch_size, sequence_length - 1, vocab_size]

            # Compute loss
            batch_size, seq_len, _ = outputs.shape
            outputs = outputs.view(-1, vocab_size)  # Shape: [batch_size * (sequence_length - 1), vocab_size]
            targets = captions_idx[:, 1:].contiguous().view(-1)  # Shape: [batch_size * (sequence_length - 1)]

            # Ensure dimensions match
            if outputs.size(0) != targets.size(0):
                raise ValueError(f"Output size {outputs.size(0)} does not match target size {targets.size(0)}")

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

# Perform pretraining
pretrain_cross_entropy(model, train_loader, resnet, device, optimizer, criterion, vocab_size, epochs=5)

# Reward function placeholder for SCST
def reward_function(predicted_caption, greedy_caption):
    return sentence_bleu([greedy_caption.split()], predicted_caption.split()), sentence_bleu([greedy_caption.split()], greedy_caption.split())

# SCST training function
def scst_training(model, train_loader, resnet, optimizer, device, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_scst_loss = 0

        for imgs, captions in train_loader:
            imgs = imgs.to(device)

            # Convert captions to indices
            captions_idx = [[vocab.get_word_index(word) for word in caption.split()] for caption in captions]
            max_len = max(len(caption) for caption in captions_idx)
            captions_idx = [caption + [vocab.get_word_index('<pad>')] * (max_len - len(caption)) for caption in captions_idx]
            captions_idx = torch.tensor(captions_idx).to(device)

            # Extract features using ResNet
            with torch.no_grad():
                features = resnet(imgs).squeeze()

            # Generate greedy captions (argmax)
            greedy_captions = [model.sample(features[i], greedy=True) for i in range(features.size(0))]

            # Sample captions with sampling strategy
            sampled_captions = [model.sample(features[i], greedy=False) for i in range(features.size(0))]

            for i in range(len(sampled_captions)):
                predicted_reward, greedy_reward = reward_function(sampled_captions[i], greedy_captions[i])
                reward_diff = predicted_reward - greedy_reward

                # Compute SCST loss
                outputs = model(features[i].unsqueeze(0), captions_idx[i:i+1, :-1])
                outputs = outputs.view(-1, vocab_size)
                targets = captions_idx[i:i+1, 1:].contiguous().view(-1)

                if outputs.size(0) != targets.size(0):
                    raise ValueError(f"Output size {outputs.size(0)} does not match target size {targets.size(0)}")

                loss = -reward_diff * torch.sum(F.log_softmax(outputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1))
                total_scst_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], SCST Loss: {total_scst_loss:.4f}")

# Perform SCST training
scst_training(model, train_loader, resnet, optimizer, device, epochs=5)

# Save the model
torch.save(model.state_dict(), "scst_image_captioning_model.pth")

# Evaluate the model by generating captions for validation images
model.eval()
with torch.no_grad():
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        features = resnet(imgs).squeeze()
        generated_captions = [model.sample(features[i], greedy=True) for i in range(features.size(0))]
        print("Generated Captions: ", generated_captions)

# Load the saved model for future use
# model.load_state_dict(torch.load("scst_image_captioning_model.pth"))
# model.eval()
