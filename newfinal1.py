import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
from pycocotools.coco import COCO
import nltk
import pickle
import numpy as np
import json
import math
from collections import Counter
import matplotlib.pyplot as plt

# Ensure nltk punkt is downloaded
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = '/home/omer/projects/code03/checkpoints'

class Vocabulary(object):
    def __init__(self, vocab_threshold, vocab_file='./vocab.pkl', start_word="<start>", end_word="<end>", unk_word="<unk>",
                 annotations_file='/mnt/c/Users/muham/Downloads/coco7/annotations/captions_train2017.json', vocab_from_file=False):
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print(f"[{i}/{len(ids)}] Tokenizing captions...")

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_word])

    def __len__(self):
        return len(self.word2idx)

class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, end_word, unk_word,
                 annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file,
                                vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in range(len(self.ids))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            return image, caption

        else:
            path = self.paths[index]

            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)

def get_loader(transform, mode='train', batch_size=1, vocab_threshold=None, vocab_file='./vocab.pkl', start_word="<start>",
               end_word="<end>", unk_word="<unk>", vocab_from_file=True, num_workers=0, cocoapi_loc='/mnt/c/Users/muham/Downloads/coco7'):
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if vocab_from_file == False:
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    if mode == 'train':
        if vocab_from_file:
            assert os.path.exists(vocab_file), "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'train2017')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_train2017.json')
    else:
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'test2017')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/image_info_test2017.json')

    dataset = CoCoDataset(transform=transform, mode=mode, batch_size=batch_size, vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file, start_word=start_word, end_word=end_word, unk_word=unk_word,
                          annotations_file=annotations_file, vocab_from_file=vocab_from_file, img_folder=img_folder)

    if mode == 'train':
        indices = dataset.get_train_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=dataset.batch_size, shuffle=True, num_workers=num_workers)

    return data_loader

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, features, captions):
        captions = captions[:, :-1]
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embedding(captions)
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.fc(lstm_out)
        return outputs

    def Predict(self, inputs, max_len=20):
        final_output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_idx = torch.max(outputs, dim=1)
            final_output.append(max_idx.cpu().numpy()[0].item())
            if (max_idx == 1 or len(final_output) >= max_len):
                break

            inputs = self.word_embedding(max_idx)
            inputs = inputs.unsqueeze(1)
        return final_output

def train(encoder, decoder, data_loader_train, num_epochs=4, print_every=150, save_every=1):
    encoder.train()
    decoder.train()

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    all_params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = optim.Adam(params=all_params, lr=lr)

    vocab_size = len(data_loader_train.dataset.vocab)
    total_step = math.ceil(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)

    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(data_loader_train):
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)

            targets = captions[:, 1:]
            outputs = outputs[:, :-1, :].reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        if (epoch + 1) % save_every == 0:
            # Save model checkpoints
            os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
            torch.save(encoder.state_dict(), os.path.join(model_save_path, f'encoderdata_{epoch + 1}.pkl'))
            torch.save(decoder.state_dict(), os.path.join(model_save_path, f'decoderdata_{epoch + 1}.pkl'))

def get_sentences(original_img, all_predictions):
    sentence = ' '
    plt.imshow(original_img.squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    return sentence.join([data_loader_test.dataset.vocab.idx2word[idx] for idx in all_predictions[1:-1]])

if __name__ == "__main__":
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Set the minimum word count threshold.
    vocab_threshold = 8

    # Specify the batch size.
    batch_size = 200

    # Define model parameters
    embed_size = 256
    hidden_size = 100
    num_layers = 1
    num_epochs = 4
    print_every = 150
    save_every = 1

    # Create DataLoader for training
    data_loader_train = get_loader(
        transform=transform_train,
        mode='train',
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_from_file=False,
        cocoapi_loc='/mnt/c/Users/muham/Downloads/coco7'
    )

    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(data_loader_train.dataset.vocab), num_layers).to(device)

    # Train models
    train(encoder, decoder, data_loader_train, num_epochs, print_every, save_every)

    # Create DataLoader for test data
    data_loader_test = get_loader(
        transform=transform_test,
        mode='test',
        cocoapi_loc='/mnt/c/Users/muham/Downloads/coco7'
    )

    # Load the saved models
    model_save_path = '/path/to/your/checkpoints'  # Update this path as necessary
    os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
    encoder.load_state_dict(torch.load(os.path.join(model_save_path, 'encoderdata_4.pkl')))
    decoder.load_state_dict(torch.load(os.path.join(model_save_path, 'decoderdata_4.pkl')))

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    # Run inference
    data_iter = iter(data_loader_test)
    original_img, processed_img = next(data_iter)

    features = encoder(processed_img.to(device)).unsqueeze(1)
    final_output = decoder.Predict(features, max_len=20)

    sentence = get_sentences(original_img, final_output)
    print("Generated Sentence:", sentence)