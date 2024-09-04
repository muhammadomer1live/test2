import torch
import torch.nn as nn
from torchvision import transforms, models
from pycocotools.coco import COCO
from torchvision.datasets import CocoCaptions
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from torch.utils.data import Subset
from PIL import Image

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

# Load the model and vocabulary
def load_model_and_vocab(model_path='model.pth', vocab_path='vocab.pth'):
    vocab = torch.load(vocab_path)
    word_to_idx = vocab['word_to_idx']
    idx_to_word = vocab['idx_to_word']
    
    cnn_model = models.resnet18(pretrained=True)
    cnn_model.fc = nn.Identity()
    
    vocab_size = len(word_to_idx)
    model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, word_to_idx, idx_to_word

# Generate captions
def generate_caption(model, image, word_to_idx, idx_to_word, max_length=20):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        caption_tokens = [word_to_idx['<START>']]
        for _ in range(max_length):
            input_tokens = torch.tensor([caption_tokens], device=device)
            outputs = model(image, input_tokens)
            predicted_id = outputs[:, -1, :].argmax(dim=-1).item()
            caption_tokens.append(predicted_id)
            if predicted_id == word_to_idx['<END>']:
                break
        return detokenize_caption(caption_tokens, idx_to_word)

# Plot example captions
def plot_example_captions(model, dataset, word_to_idx, idx_to_word, num_examples=5, max_length=20):
    model.eval()
    fig, axes = plt.subplots(num_examples, 1, figsize=(8, num_examples * 3))
    
    if num_examples == 1:
        axes = [axes]
    
    for i in range(num_examples):
        image, caption = dataset[i]
        predicted_caption = generate_caption(model, image, word_to_idx, idx_to_word, max_length)
        
        axes[i].imshow(image.permute(1, 2, 0).cpu())
        axes[i].set_title(f"Predicted: {predicted_caption}\nOriginal: {caption[0]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    device = 'cpu'

    # Load the model and vocabulary
    model, word_to_idx, idx_to_word = load_model_and_vocab(
        model_path='model.pth',
        vocab_path='vocab.pth'
    )
    
    model = model.to(device)

    # Load the dataset
    coco_root = '/mnt/c/Users/muham/Downloads/coco7/'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = CocoCaptions(
        root=f"{coco_root}/train2017",
        annFile=f"{coco_root}/annotations/captions_train2017.json",
        transform=transform
    )

    # Create a subset of the dataset for quick testing
    def create_subset(dataset, max_size=100):
        indices = np.random.choice(len(dataset), max_size, replace=False)
        indices = list(map(int, indices))
        return Subset(dataset, indices)

    train_dataset = create_subset(train_dataset, max_size=100)

    # Plot example captions
    plot_example_captions(model, train_dataset, word_to_idx, idx_to_word, num_examples=5)
