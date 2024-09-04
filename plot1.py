import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats

# Dummy values for NIC, SCST, and SAT
nic_rouge_scores = {'rouge1': 0.30, 'rouge2': 0.15, 'rougeL': 0.25}
scst_rouge_scores = {'rouge1': 0.32, 'rouge2': 0.17, 'rougeL': 0.27}
sat_rouge_scores = {'rouge1': 0.28, 'rouge2': 0.14, 'rougeL': 0.24}

nic_bleu_score = 0.20
scst_bleu_score = 0.22
sat_bleu_score = 0.21

nic_caption_lengths = [10, 12, 15, 18, 20, 25, 30] * 50
scst_caption_lengths = [11, 13, 16, 19, 22, 28, 35] * 50
sat_caption_lengths = [12, 14, 17, 19, 21, 26, 31] * 50

captions_nic = ['a cat on the mat', 'a cat sitting on a mat', 'a small cat on the mat'] * 50
captions_scst = ['a dog on the rug', 'a dog lying on the rug', 'a large dog on the rug'] * 50
captions_sat = ['a cat resting on a mat', 'a small cat lying on a mat', 'a cat is on the mat'] * 50

# Dummy embeddings
vocab = ["cat", "dog", "mat", "rug", "sitting", "lying", "resting", "small", "large"]
nic_word_embeddings = np.random.rand(9, 50)
scst_word_embeddings = np.random.rand(9, 50)
sat_word_embeddings = np.random.rand(9, 50)

# Dummy loss values for each model
nic_train_losses = [0.8, 0.7, 0.6, 0.5, 0.4]
nic_val_losses = [0.85, 0.75, 0.7, 0.65, 0.6]

scst_train_losses = [0.78, 0.68, 0.58, 0.48, 0.38]
scst_val_losses = [0.83, 0.73, 0.68, 0.63, 0.58]

sat_train_losses = [0.76, 0.66, 0.56, 0.46, 0.36]
sat_val_losses = [0.80, 0.70, 0.65, 0.60, 0.55]

# Visualization functions

def plot_rouge_and_bleu_scores():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROUGE Scores
    ax[0].bar(nic_rouge_scores.keys(), nic_rouge_scores.values(), color='b', alpha=0.5, label='NIC')
    ax[0].bar(scst_rouge_scores.keys(), scst_rouge_scores.values(), color='r', alpha=0.5, label='SCST')
    ax[0].bar(sat_rouge_scores.keys(), sat_rouge_scores.values(), color='g', alpha=0.5, label='SAT')
    ax[0].set_title('ROUGE Scores Comparison')
    ax[0].set_xlabel('ROUGE Metric')
    ax[0].set_ylabel('Score')
    ax[0].legend()

    # Plot BLEU Scores
    ax[1].bar(['NIC', 'SCST', 'SAT'], [nic_bleu_score, scst_bleu_score, sat_bleu_score], color=['b', 'r', 'g'])
    ax[1].set_title('BLEU Score Comparison')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('BLEU Score')

    plt.tight_layout()
    return fig

def plot_caption_lengths():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(nic_caption_lengths, bins=range(0, max(nic_caption_lengths)+1), alpha=0.7, label='NIC Caption Lengths')
    ax.hist(scst_caption_lengths, bins=range(0, max(scst_caption_lengths)+1), alpha=0.7, label='SCST Caption Lengths')
    ax.hist(sat_caption_lengths, bins=range(0, max(sat_caption_lengths)+1), alpha=0.7, label='SAT Caption Lengths')
    ax.set_title('Caption Length Distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = np.arange(len(nic_train_losses))  # Assuming all models have the same number of epochs

    # Plot Training Losses
    ax.plot(epochs, nic_train_losses, label='NIC Training Loss', color='b', linestyle='--')
    ax.plot(epochs, scst_train_losses, label='SCST Training Loss', color='r', linestyle='--')
    ax.plot(epochs, sat_train_losses, label='SAT Training Loss', color='g', linestyle='--')

    # Plot Validation Losses
    ax.plot(epochs, nic_val_losses, label='NIC Validation Loss', color='b')
    ax.plot(epochs, scst_val_losses, label='SCST Validation Loss', color='r')
    ax.plot(epochs, sat_val_losses, label='SAT Validation Loss', color='g')

    ax.set_title('Learning Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig

def plot_caption_diversity(captions_list, model_name):
    word_counts = Counter()
    for caption in captions_list:
        words = caption.split()
        word_counts.update(words)
        
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(word_counts.keys(), word_counts.values(), color='b')
    ax.set_title(f'Vocabulary Diversity for {model_name}')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=90)
    return fig

def plot_pca_embeddings():
    pca = PCA(n_components=2)
    nic_pca = pca.fit_transform(nic_word_embeddings)
    scst_pca = pca.fit_transform(scst_word_embeddings)
    sat_pca = pca.fit_transform(sat_word_embeddings)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(nic_pca[:, 0], nic_pca[:, 1], c='b', label="NIC Embeddings")
    ax.scatter(scst_pca[:, 0], scst_pca[:, 1], c='r', label="SCST Embeddings")
    ax.scatter(sat_pca[:, 0], sat_pca[:, 1], c='g', label="SAT Embeddings")
    for i, word in enumerate(vocab):
        ax.annotate(word, (nic_pca[i, 0], nic_pca[i, 1]))
    ax.set_title('PCA on Word Embeddings')
    ax.legend()
    return fig

def plot_correlation_heatmap():
    data = {
        'BLEU': [nic_bleu_score, scst_bleu_score, sat_bleu_score],
        'ROUGE-1': [nic_rouge_scores['rouge1'], scst_rouge_scores['rouge1'], sat_rouge_scores['rouge1']],
        'ROUGE-2': [nic_rouge_scores['rouge2'], scst_rouge_scores['rouge2'], sat_rouge_scores['rouge2']],
        'ROUGE-L': [nic_rouge_scores['rougeL'], scst_rouge_scores['rougeL'], sat_rouge_scores['rougeL']]
    }
    correlation_matrix = np.corrcoef(list(data.values()))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, xticklabels=data.keys(), yticklabels=data.keys(), ax=ax)
    ax.set_title("Correlation Heatmap Between BLEU and ROUGE Scores")
    return fig

def plot_kmeans_clustering():
    kmeans = KMeans(n_clusters=3)
    caption_lengths = np.array(nic_caption_lengths + scst_caption_lengths + sat_caption_lengths).reshape(-1, 1)
    clusters = kmeans.fit_predict(caption_lengths)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(np.arange(len(caption_lengths)), caption_lengths, c=clusters, cmap='viridis')
    ax.set_title('KMeans Clustering on Caption Lengths')
    ax.set_xlabel('Caption Index')
    ax.set_ylabel('Caption Length')
    return fig

def plot_linear_regression():
    X = np.arange(len(nic_train_losses)).reshape(-1, 1)
    y = np.array(nic_train_losses)
    reg = LinearRegression().fit(X, y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X, y, color='blue', label='Train Loss')
    ax.plot(X, reg.predict(X), color='red', label='Linear Fit')
    ax.set_title('Linear Regression on Learning Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.legend()
    return fig

def plot_t_test():
    t_stat, p_value = stats.ttest_ind([nic_bleu_score, sat_bleu_score], [scst_bleu_score])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['NIC + SAT', 'SCST'], [nic_bleu_score, scst_bleu_score], color=['blue', 'red'])
    ax.set_title(f'T-test BLEU Scores (p-value: {p_value:.4f})')
    return fig

class PlotViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plot Viewer")

        self.plot_funcs = [
            ("ROUGE and BLEU Scores", plot_rouge_and_bleu_scores),
            ("Caption Lengths", plot_caption_lengths),
            ("Learning Curves", plot_learning_curves),
            ("NIC Caption Diversity", lambda: plot_caption_diversity(captions_nic, 'NIC')),
            ("SCST Caption Diversity", lambda: plot_caption_diversity(captions_scst, 'SCST')),
            ("SAT Caption Diversity", lambda: plot_caption_diversity(captions_sat, 'SAT')),
            ("PCA on Word Embeddings", plot_pca_embeddings),
            ("Correlation Heatmap", plot_correlation_heatmap),
            ("KMeans Clustering", plot_kmeans_clustering),
            ("Linear Regression", plot_linear_regression),
            ("T-Test on BLEU Scores", plot_t_test)
        ]
        self.current_plot = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.show_prev_plot)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.show_plot(self.current_plot)

    def show_plot(self, index):
        self.fig.clear()
        plot_name, plot_func = self.plot_funcs[index]
        fig = plot_func()
        self.canvas.figure = fig
        self.canvas.draw()

    def show_prev_plot(self):
        self.current_plot = (self.current_plot - 1) % len(self.plot_funcs)
        self.show_plot(self.current_plot)

    def show_next_plot(self):
        self.current_plot = (self.current_plot + 1) % len(self.plot_funcs)
        self.show_plot(self.current_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotViewerApp(root)
    root.mainloop()
