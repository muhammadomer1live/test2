# model_utils.py

import torch
import matplotlib.pyplot as plt
from PIL import Image

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def save_vocab(word_to_idx, idx_to_word, vocab_path):
    vocab = {'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word}
    torch.save(vocab, vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

def load_vocab(vocab_path):
    vocab = torch.load(vocab_path)
    return vocab['word_to_idx'], vocab['idx_to_word']

def generate_caption(model, image, word_to_idx, idx_to_word, max_length=20, device='cpu'):
    model.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        
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
        caption = detokenize_caption(caption_tokens, idx_to_word)
    return caption

def display_image_with_caption(image_path, model, word_to_idx, idx_to_word, transform, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    caption = generate_caption(model, image, word_to_idx, idx_to_word, device=device)
    
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Generated Caption: {caption}")
    plt.axis('off')
    plt.show()

def detokenize_caption(tokens, idx_to_word):
    caption = []
    for token in tokens:
        word = idx_to_word.get(token, '')
        if word in ['<PAD>', '<START>', '<END>', '<UNK>']:
            continue
        caption.append(word)
    return ' '.join(caption)
