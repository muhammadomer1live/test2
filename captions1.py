# main.py

import torch
from torchvision import transforms, models
from captions2 import load_model, load_vocab, display_image_with_caption
#from try11 import cnn_model, SCSTModel, ImageCaptioningModel

# Define your device
device = 'cpu'

# Load vocabularies
nic_word_to_idx, nic_idx_to_word = load_vocab('nic_vocab.pth')
scst_word_to_idx, scst_idx_to_word = load_vocab('scst_vocab.pth')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load NIC model
#nic_model = ImageCaptioningModel(cnn_model, len(nic_word_to_idx), embedding_dim=256, hidden_dim=256)
nic_model = load_model('nic_model.pth')

# Load SCST model
#scst_model = SCSTModel(cnn_model, len(scst_word_to_idx), embedding_dim=256, hidden_dim=256)
scst_model = load_model('scst_model.pth')

# Example usage for NIC model
image_path = '/mnt/c/Users/muham/Downloads/coco7/1.jpg'
print("NIC Model:")
display_image_with_caption(image_path, nic_model, nic_word_to_idx, nic_idx_to_word, transform, device=device)

# Example usage for SCST model
print("SCST Model:")
display_image_with_caption(image_path, scst_model, scst_word_to_idx, scst_idx_to_word, transform, device=device)
