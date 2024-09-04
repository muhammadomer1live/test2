import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def convert_to_original_format(data):
    """
    Recursively converts data from JSON serializable format back to its original format.
    """
    if isinstance(data, dict):
        return {k: convert_to_original_format(v) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            # Recursively convert nested lists
            return [convert_to_original_format(i) for i in data]
        elif all(isinstance(i, (int, float)) for i in data):
            # Convert lists of numbers to numpy arrays
            return np.array(data, dtype=np.float64)
        else:
            # Convert lists of mixed types
            return [convert_to_original_format(i) for i in data]
    elif isinstance(data, (int, float)):
        # Convert individual numbers to appropriate numpy types
        return np.int64(data) if isinstance(data, int) else np.float64(data)
    else:
        return data

def load_metrics_from_json(filename):
    # Load the data from the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert data back to original format
    original_data = convert_to_original_format(data)
    
    return original_data

# Load metrics
metrics = load_metrics_from_json('metrics2.json')

# Access the metrics
epoch_losses = metrics.get('epoch_losses', [])
val_losses = metrics.get('val_losses', [])
nic_rouge_scores = metrics.get('nic_rouge_scores', [])
scst_rouge_scores = metrics.get('scst_rouge_scores', [])
nic_bleu_score = metrics.get('nic_bleu_score', [])
scst_bleu_score = metrics.get('scst_bleu_score', [])
nic_caption_lengths = metrics.get('nic_caption_lengths', [])
scst_caption_lengths = metrics.get('scst_caption_lengths', [])
captions_nic = metrics.get('captions_nic', [])
captions_scst = metrics.get('captions_scst', [])

# ROUGE and BLEU Scores Comparison Visualization
def plot_rouge_and_bleu_scores(rouge_scores_list, bleu_scores_list):
    plt.figure(figsize=(12, 6))

    # ROUGE Scores
    for i, rouge_scores in enumerate(rouge_scores_list):
        plt.plot([1, 2, 3], [rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL']],
                 label=f'ROUGE Scores - Set {i+1}', marker='o')
    
    # BLEU Scores
    plt.plot(range(len(bleu_scores_list)), bleu_scores_list, label='BLEU Scores', marker='o', linestyle='--')

    plt.title("ROUGE and BLEU Scores Comparison")
    plt.ylabel("Scores")
    plt.xlabel("Metrics")
    plt.legend()
    plt.xticks(ticks=[1, 2, 3], labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    plt.show()

# Caption Length Distribution Visualization
def plot_caption_length_distribution(caption_lengths, title):
    # Flatten the list of caption lengths if nested
    flat_lengths = [length for sublist in caption_lengths for length in sublist]

    plt.figure(figsize=(8, 5))
    sns.histplot(flat_lengths, bins=20, kde=True)
    plt.title(title)
    plt.xlabel("Caption Length")
    plt.ylabel("Frequency")
    plt.show()

# Learning Curves Visualization
def plot_learning_curves(epoch_losses, val_losses):
    plt.figure(figsize=(12, 6))
    
    # Plot training losses
    plt.plot(epoch_losses, label="Training Loss", color='blue', linestyle='-', marker='o')

    # Plot validation losses
    plt.plot(val_losses, label="Validation Loss", color='orange', linestyle='--', marker='x')
    
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
# Plot ROUGE and BLEU Scores
plot_rouge_and_bleu_scores(nic_rouge_scores, nic_bleu_score)

# Plot Caption Length Distribution
plot_caption_length_distribution(nic_caption_lengths, "NIC Caption Length Distribution")
plot_caption_length_distribution(scst_caption_lengths, "SCST Caption Length Distribution")

# Plot Learning Curves
plot_learning_curves(epoch_losses, val_losses)
