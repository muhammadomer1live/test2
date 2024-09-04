import torch
import random
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from final7 import ImageCaptioningModel, build_coco_vocab, detokenize_caption, test_image_captioning

# Define paths
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'
model_path = 'nic_model.pth'  # Path to the saved NIC model
annotations_file = f"{coco_root}/annotations/captions_val2017.json"  # Use validation set for test

# Initialize COCO
coco = COCO(annotations_file)

# Build vocabulary
word_to_idx, idx_to_word = build_coco_vocab(coco)

# Load the NIC model
device = 'cpu'
cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
cnn_model.fc = torch.nn.Identity()
vocab_size = len(word_to_idx)
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
nic_model.load_state_dict(torch.load(model_path))
nic_model.to(device)

# Load a random image from the dataset
image_ids = coco.getImgIds()
random_image_id = random.choice(image_ids)
image_info = coco.loadImgs(random_image_id)[0]
image_path = f"{coco_root}/val2017/{image_info['file_name']}"

# Generate a caption for the image
caption = test_image_captioning(nic_model, image_path, word_to_idx, idx_to_word)

# Display the image and captions
def show_image_and_caption(image_path, predicted_caption):
    image = Image.open(image_path)
    
    # Plot image and caption
    plt.figure(figsize=(10, 5))
    
    # Show the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image")
    
    # Show the predicted caption
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"Predicted Caption:\n{predicted_caption}", fontsize=12, ha='center', va='center')
    plt.axis('off')
    plt.title("Generated Caption")
    
    plt.show()

# Display the image and the caption
show_image_and_caption(image_path, caption)
