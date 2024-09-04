import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import Counter
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

# Dummy values for NIC, SCST, and SAT models
nic_rouge_scores = {'rouge1': 0.30, 'rouge2': 0.15, 'rougeL': 0.25}
scst_rouge_scores = {'rouge1': 0.32, 'rouge2': 0.17, 'rougeL': 0.27}
sat_rouge_scores = {'rouge1': 0.31, 'rouge2': 0.16, 'rougeL': 0.26}

nic_bleu_score = 0.20
scst_bleu_score = 0.22
sat_bleu_score = 0.21

nic_caption_lengths = np.random.randint(8, 35, 300)
scst_caption_lengths = np.random.randint(9, 40, 300)
sat_caption_lengths = np.random.randint(10, 45, 300)

train_losses = np.random.uniform(0.4, 0.8, 100).tolist()
val_losses = np.random.uniform(0.4, 0.8, 100).tolist()
sat_train_losses = np.random.uniform(0.3, 0.7, 100).tolist()
sat_val_losses = np.random.uniform(0.3, 0.7, 100).tolist()

captions_nic = ['a cat on the mat', 'a cat sitting on a mat', 'a small cat on the mat'] * 100
captions_scst = ['a dog on the rug', 'a dog lying on the rug', 'a large dog on the rug'] * 100
sat_captions = ['a person in a park', 'a person sitting on a bench', 'a person walking in the park'] * 100

class PlotViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plot Viewer")

        self.plot_funcs = [
            ("ROUGE and BLEU Scores", self.plot_rouge_and_bleu_scores),
            ("Caption Lengths", self.plot_caption_lengths),
            ("Learning Curves", self.plot_learning_curves),
            ("NIC Caption Diversity", lambda: self.plot_caption_diversity(captions_nic, 'NIC')),
            ("SCST Caption Diversity", lambda: self.plot_caption_diversity(captions_scst, 'SCST')),
            ("SAT Caption Diversity", lambda: self.plot_caption_diversity(sat_captions, 'SAT')),
            ("PCA on Word Embeddings", self.plot_pca_on_word_embeddings),
            ("Correlation Heatmap", self.plot_correlation_heatmap),
            ("KMeans Clustering", self.plot_kmeans_clustering),
            ("Linear Regression on Learning Curves", self.plot_linear_regression_on_learning_curves),
            ("BLEU Score T-Test", self.plot_bleu_score_t_test)
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
        plot_func()
        self.canvas.draw()

    def show_prev_plot(self):
        self.current_plot = (self.current_plot - 1) % len(self.plot_funcs)
        self.show_plot(self.current_plot)

    def show_next_plot(self):
        self.current_plot = (self.current_plot + 1) % len(self.plot_funcs)
        self.show_plot(self.current_plot)

    def plot_rouge_and_bleu_scores(self):
        ax = self.fig.add_subplot(111)
        models = ['NIC', 'SCST', 'SAT']
        rouge1_scores = [nic_rouge_scores['rouge1'], scst_rouge_scores['rouge1'], sat_rouge_scores['rouge1']]
        rouge2_scores = [nic_rouge_scores['rouge2'], scst_rouge_scores['rouge2'], sat_rouge_scores['rouge2']]
        rougeL_scores = [nic_rouge_scores['rougeL'], scst_rouge_scores['rougeL'], sat_rouge_scores['rougeL']]
        bleu_scores = [nic_bleu_score, scst_bleu_score, sat_bleu_score]

        bar_width = 0.2
        index = np.arange(len(models))
        ax.bar(index - bar_width, rouge1_scores, bar_width, label='ROUGE-1')
        ax.bar(index, rouge2_scores, bar_width, label='ROUGE-2')
        ax.bar(index + bar_width, rougeL_scores, bar_width, label='ROUGE-L')
        ax.set_xticks(index)
        ax.set_xticklabels(models)
        ax.set_title('ROUGE Scores Comparison')
        ax.set_ylabel('Score')
        ax.legend()

        ax2 = ax.twinx()
        ax2.plot(index, bleu_scores, 'o--', color='black', label='BLEU Score')
        ax2.set_ylabel('BLEU Score')
        ax2.legend(loc='upper right')

    def plot_caption_lengths(self):
        ax = self.fig.add_subplot(111)
        ax.hist(nic_caption_lengths, bins=range(0, max(nic_caption_lengths) + 1), alpha=0.5, label='NIC Caption Lengths')
        ax.hist(scst_caption_lengths, bins=range(0, max(scst_caption_lengths) + 1), alpha=0.5, label='SCST Caption Lengths')
        ax.hist(sat_caption_lengths, bins=range(0, max(sat_caption_lengths) + 1), alpha=0.5, label='SAT Caption Lengths')
        ax.set_title('Caption Length Distribution')
        ax.set_xlabel('Length')
        ax.set_ylabel('Frequency')
        ax.legend()

    def plot_learning_curves(self):
        ax = self.fig.add_subplot(111)
        ax.plot(train_losses, label='Training Loss (NIC)')
        ax.plot(val_losses, label='Validation Loss (NIC)')
        ax.plot(sat_train_losses, label='Training Loss (SAT)', linestyle='--')
        ax.plot(sat_val_losses, label='Validation Loss (SAT)', linestyle='--')
        ax.set_title('Learning Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    def plot_caption_diversity(self, captions_list, model_name):
        word_counts = Counter()
        for caption in captions_list:
            words = caption.split()
            word_counts.update(words)
        
        ax = self.fig.add_subplot(111)
        ax.bar(word_counts.keys(), word_counts.values(), color='b')
        ax.set_title(f'Vocabulary Diversity for {model_name}')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=90)

    def plot_pca_on_word_embeddings(self):
        num_words = 500
        np.random.seed(0)
        embeddings = np.random.randn(num_words, 50)
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        ax = self.fig.add_subplot(111)
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        ax.set_title('PCA on Word Embeddings')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')

    def plot_correlation_heatmap(self):
        data = {
            'ROUGE-1': [nic_rouge_scores['rouge1'], scst_rouge_scores['rouge1'], sat_rouge_scores['rouge1']],
            'ROUGE-2': [nic_rouge_scores['rouge2'], scst_rouge_scores['rouge2'], sat_rouge_scores['rouge2']],
            'ROUGE-L': [nic_rouge_scores['rougeL'], scst_rouge_scores['rougeL'], sat_rouge_scores['rougeL']],
            'BLEU': [nic_bleu_score, scst_bleu_score, sat_bleu_score]
        }
        
        df = pd.DataFrame(data, index=['NIC', 'SCST', 'SAT'])
        correlation = df.corr()

        ax = self.fig.add_subplot(111)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')

    def plot_kmeans_clustering(self):
        caption_lengths = np.array([nic_caption_lengths, scst_caption_lengths, sat_caption_lengths]).flatten()
        labels = ['NIC'] * len(nic_caption_lengths) + ['SCST'] * len(scst_caption_lengths) + ['SAT'] * len(sat_caption_lengths)
        
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(caption_lengths.reshape(-1, 1))
        
        ax = self.fig.add_subplot(111)
        scatter = ax.scatter(caption_lengths, clusters, c=clusters, cmap='viridis')
        ax.set_title('KMeans Clustering of Caption Lengths')
        ax.set_xlabel('Caption Length')
        ax.set_ylabel('Cluster')
        ax.legend(handles=scatter.legend_elements()[0], labels=['NIC', 'SCST', 'SAT'])

    def plot_linear_regression_on_learning_curves(self):
        epochs = np.arange(len(train_losses))
        reg_nic = LinearRegression().fit(epochs.reshape(-1, 1), train_losses)
        reg_sat = LinearRegression().fit(epochs.reshape(-1, 1), sat_train_losses)
        
        r_nic_train_pred = reg_nic.predict(epochs.reshape(-1, 1))
        r_sat_train_pred = reg_sat.predict(epochs.reshape(-1, 1))
        
        ax = self.fig.add_subplot(111)
        ax.plot(train_losses, label='Training Loss (NIC)')
        ax.plot(sat_train_losses, label='Training Loss (SAT)')
        ax.plot(r_nic_train_pred, label='Linear Regression (NIC)', linestyle='--')
        ax.plot(r_sat_train_pred, label='Linear Regression (SAT)', linestyle='--')
        ax.set_title('Linear Regression on Learning Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    def plot_bleu_score_t_test(self):
        bleu_scores = {
            'NIC': [nic_bleu_score],
            'SCST': [scst_bleu_score],
            'SAT': [sat_bleu_score]
        }
        
        t_stat_nic_scst, p_val_nic_scst = ttest_ind(bleu_scores['NIC'], bleu_scores['SCST'])
        t_stat_nic_sat, p_val_nic_sat = ttest_ind(bleu_scores['NIC'], bleu_scores['SAT'])
        t_stat_scst_sat, p_val_scst_sat = ttest_ind(bleu_scores['SCST'], bleu_scores['SAT'])
        
        ax = self.fig.add_subplot(111)
        ax.text(0.1, 0.9, f'Test NIC vs SCST: t={t_stat_nic_scst:.2f}, p={p_val_nic_scst:.3f}', fontsize=12)
        ax.text(0.1, 0.7, f'Test NIC vs SAT: t={t_stat_nic_sat:.2f}, p={p_val_nic_sat:.3f}', fontsize=12)
        ax.text(0.1, 0.5, f'Test SCST vs SAT: t={t_stat_scst_sat:.2f}, p={p_val_scst_sat:.3f}', fontsize=12)
        ax.axis('off')
        ax.set_title('BLEU Score T-Test Results')

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotViewerApp(root)
    root.mainloop()
