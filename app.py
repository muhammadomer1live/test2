import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np

# Define the ImageCaptioningModel class with the correct vocabulary size
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

# Define functions to handle model and vocabulary loading
def load_model_and_vocab(model_path, vocab_path):
    # Load model
    cnn_model = models.resnet18(weights='DEFAULT')
    cnn_model.fc = nn.Identity()
    
    # Use the vocabulary size that matches the checkpoint
    vocab_size = 14030  # Update this to the correct size from the checkpoint
    
    model = ImageCaptioningModel(cnn_model, vocab_size=vocab_size, embedding_dim=256, hidden_dim=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = f.read().splitlines()
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return model, word_to_idx, idx_to_word

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define the greedy search function for generating captions
def greedy_search(model, image, word_to_idx, idx_to_word, max_length=20):
    device = 'cpu'
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

        caption = caption.squeeze().cpu()
        return detokenize_caption(caption, idx_to_word)

def detokenize_caption(tokens, idx_to_word):
    return ' '.join(idx_to_word.get(token, '') for token in tokens if token not in [word_to_idx['<PAD>'], word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<UNK>']])

# Create the GUI application
class ImageCaptioningApp:
    def __init__(self, root, model, word_to_idx, idx_to_word):
        self.root = root
        self.model = model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.image_label = None
        
        root.title("Image Captioning App")
        
        # Create a file upload button
        upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        upload_button.pack()
        
        # Create a label to display the image
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        # Create a label to display the caption
        self.caption_label = tk.Label(root, text="", wraplength=400)
        self.caption_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = transform(image).unsqueeze(0)  # Apply transformation and add batch dimension
            
            caption = greedy_search(self.model, image.squeeze(0), self.word_to_idx, self.idx_to_word)
            self.display_image(file_path)
            self.caption_label.config(text=f"Generated Caption: {caption}")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo

# Main application code
if __name__ == "__main__":
    root = tk.Tk()
    model, word_to_idx, idx_to_word = load_model_and_vocab('/mnt/c/Users/muham/Downloads/coco7/image_captioning_model_scst.pth', '/mnt/c/Users/muham/Downloads/coco7/vocab.txt')
    app = ImageCaptioningApp(root, model, word_to_idx, idx_to_word)
    root.mainloop()
