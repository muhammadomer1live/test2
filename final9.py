import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from final8 import transform, device, idx_to_word, val_dataloader, nic_model

# Function to display an image with captions
def display_image_and_captions(image, original_caption, predicted_caption):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.title(f"Original Caption: {original_caption}\nPredicted Caption: {predicted_caption}")
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
