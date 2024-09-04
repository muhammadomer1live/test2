import tkinter as tk
from tkinter import scrolledtext, ttk
import tkinter.font as tkFont
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pycocotools.coco import COCO
from rouge_score import rouge_scorer
from tqdm import tqdm
import threading

# Define paths
coco_annotations_path = '/mnt/c/Users/muham/Downloads/coco7/annotations/captions_val2017.json'
coco_images_path = '/mnt/c/Users/muham/Downloads/coco7/val2017'
generated_captions_path = 'formatted_captions.json'
model_path = 'nic_model.pth'
vocab_path = 'nic_vocab.pth'

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def generate_caption_batch(model, images, word_to_idx, idx_to_word, max_length=20):
    model.eval()
    captions = []
    with torch.no_grad():
        for image in images:
            tokens = [word_to_idx['<START>']]
            for _ in range(max_length):
                inputs = torch.tensor([tokens], device=device)
                outputs = model(image.unsqueeze(0).to(device), inputs)
                pred_id = outputs[:, -1, :].argmax(dim=-1).item()
                tokens.append(pred_id)
                if pred_id == word_to_idx['<END>']:
                    break
            captions.append(detokenize_caption(tokens, idx_to_word))
    return captions

def detokenize_caption(tokens, idx_to_word):
    return ' '.join([idx_to_word.get(t, '') for t in tokens if idx_to_word.get(t, '') not in ['<PAD>', '<START>', '<END>', '<UNK>']])

import torchvision.models as models
from torchvision.models import ResNet18_Weights

def load_model_and_vocab(model_path, vocab_path):
    class ImageCaptioningModel(nn.Module):
        def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256):
            super().__init__()
            self.cnn = cnn_model
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, images, captions):
            features = self.cnn(images).unsqueeze(1)
            captions_embed = self.embedding(captions)
            inputs = torch.cat((features.repeat(1, captions_embed.size(1), 1), captions_embed), dim=2)
            inputs = self.fc_input(inputs)
            outputs, _ = self.rnn(inputs)
            return self.fc(outputs)

    # Use the new way to load ResNet18 with pre-trained weights
    cnn_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    cnn_model.fc = nn.Identity()

    # Load vocabulary
    vocab = torch.load(vocab_path, map_location='cpu', weights_only=True)
    word_to_idx = vocab['word_to_idx']
    idx_to_word = vocab['idx_to_word']
    vocab_size = len(word_to_idx)

    # Load model weights
    model = ImageCaptioningModel(cnn_model, vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.to(device)
    
    return model, word_to_idx, idx_to_word

class ProgressApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Caption Generation Progress")
        self.custom_font = tkFont.Font(family="Helvetica", size=40)
        
        self.progress_label = tk.Label(root, text="Progress:", font=self.custom_font)
        self.progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=600)
        self.progress_bar.pack(pady=20)
        
        self.results_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=30, width=100, font=self.custom_font)
        self.results_text.pack(padx=20, pady=20)
        
        self.total_batches = self.current_batch = 0

    def update_progress(self, message, progress=None):
        self.progress_label.config(text=message)
        if progress is not None:
            self.progress_bar['value'] = progress
            self.root.update_idletasks()

    def append_result(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.yview(tk.END)
        self.root.update_idletasks()

    def set_total_batches(self, total_batches):
        self.total_batches = total_batches
        self.progress_bar['maximum'] = total_batches

    def increment_batch(self):
        self.current_batch += 1
        self.update_progress(f"Processing batch {self.current_batch}/{self.total_batches}", self.current_batch)

def run_captioning_task(app):
    coco_val = COCO(coco_annotations_path)
    image_ids = coco_val.getImgIds()[:100]

    model, word_to_idx, idx_to_word = load_model_and_vocab(model_path, vocab_path)
    generated_captions = []
    batch_size = 32
    total_batches = (len(image_ids) + batch_size - 1) // batch_size
    app.set_total_batches(total_batches)

    for i in tqdm(range(0, len(image_ids), batch_size), desc="Generating Captions"):
        app.increment_batch()
        batch_ids = image_ids[i:i+batch_size]
        batch_images = [transform(Image.open(os.path.join(coco_images_path, coco_val.loadImgs(img_id)[0]['file_name'])).convert('RGB')) for img_id in batch_ids]
        batch_images = torch.stack(batch_images)
        captions = generate_caption_batch(model, batch_images, word_to_idx, idx_to_word)
        
        for img_id, caption in zip(batch_ids, captions):
            generated_captions.append({'image_id': img_id, 'caption': caption})

    with open(generated_captions_path, 'w') as f:
        json.dump(generated_captions, f)

    app.update_progress("Generating captions completed!")

    hypotheses, references = [], []
    for img_id in tqdm(image_ids, desc="Collecting Hypotheses and References"):
        hyp_caption = [item['caption'] for item in generated_captions if item['image_id'] == img_id]
        ref_captions = [ann['caption'] for ann in coco_val.loadAnns(coco_val.getAnnIds(imgIds=img_id))]
        
        if hyp_caption and ref_captions:
            hypotheses.append(hyp_caption[0])
            references.append(ref_captions)

    def calculate_rouge(hypotheses, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}
        for hyp, refs in tqdm(zip(hypotheses, references), desc="Calculating ROUGE Scores"):
            for ref in refs:
                score = scorer.score(ref, hyp)
                for key in scores:
                    scores[key].append(score[key].fmeasure)
        return {key: np.mean(val) for key, val in scores.items()}

    rouge_scores = calculate_rouge(hypotheses, references)
    app.append_result("ROUGE Scores:")
    for k, v in rouge_scores.items():
        app.append_result(f"{k}: {v:.4f}")

def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    root = tk.Tk()
    app = ProgressApp(root)
    
    threading.Thread(target=run_captioning_task, args=(app,), daemon=True).start()
    
    root.mainloop()

if __name__ == "__main__":
    main()
