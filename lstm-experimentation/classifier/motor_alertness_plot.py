import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def plot_sequences(features_path, labels_path):
    # Load the saved features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    categories = ['absent', 'twitching', 'walking']
    colors = ['blue', 'red', 'green']
    
    for perplexity in range(1,51):  # Iterate from 0 to 50
        print(f"Performing t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=float(perplexity))
        features_2d = tsne.fit_transform(features)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        for i, category in enumerate(categories):
            mask = labels == category
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                        c=colors[i], label=category, alpha=0.6)
        
        plt.title(f't-SNE Visualization of Sequence Features (Perplexity={perplexity})')
        plt.legend()
        plt.savefig(f'sequence_features_tsne_p{perplexity}.png')
        plt.close()
    
    print(f"\nTotal sequences: {len(features)}")
    for category in categories:
        count = sum(1 for label in labels if label == category)
        print(f"{category}: {count} sequences")

if __name__ == '__main__':
    parser = ArgumentParser(description="Plot t-SNE visualization from saved sequence features.")
    parser.add_argument('--features', type=str, default='all_features.npy',
                        help='Path to the saved features .npy file')
    parser.add_argument('--labels', type=str, default='all_labels.npy',
                        help='Path to the saved labels .npy file')
    
    args = parser.parse_args()
    plot_sequences(args.features, args.labels)
    print("\nAll visualizations saved with filenames 'sequence_features_tsne_pX.png'")
