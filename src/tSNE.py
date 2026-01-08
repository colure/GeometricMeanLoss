import collections
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from torch import distributed as dist


class TSNEVisualizer:
    """Visualize model embeddings using t-SNE dimensionality reduction."""

    def __init__(self, val_loader, model, args):
        self.val_loader = val_loader
        self.model = model
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Configurable parameters with sensible defaults
        self.samples_per_class = getattr(args, "tsne_samples_per_class", 150)
        self.perplexity = getattr(args, "tsne_perplexity", 60)
        self.learning_rate = getattr(args, "tsne_learning_rate", 20)
        self.n_iter = getattr(args, "tsne_n_iter", 1500)
        self.early_exaggeration = getattr(args, "tsne_early_exaggeration", 15)
        self.use_outlier_filter = getattr(args, "tsne_filter_outliers", True)
        self.lof_n_neighbors = getattr(args, "tsne_lof_neighbors", 60)

        # Output directory
        output_dir = getattr(args, "output_dir", "tsne")
        self.output_dir = os.path.join(output_dir, "tsne") if output_dir else "tsne"
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_features(self):
        """Extract embeddings from the model for all validation samples."""
        self.model.eval()
        output_dict = collections.defaultdict(list)

        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.args.device)
            outputs = self.model(inputs)[0].cpu().data.numpy()
            labels = labels.numpy()

            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        return output_dict

    def _sample_features(self, output_dict):
        """Sample a fixed number of features per class for visualization."""
        features, labels = [], []

        for label, feats in output_dict.items():
            # Sample with replacement if not enough samples
            n_samples = min(len(feats), self.samples_per_class)
            replace = n_samples < self.samples_per_class

            np.random.seed(42)  # Fixed seed for reproducibility
            indices = np.random.choice(
                len(feats), self.samples_per_class, replace=replace
            )

            features.extend([feats[i] for i in indices])
            labels.extend([label] * self.samples_per_class)

        return np.array(features), np.array(labels)

    def _filter_outliers(self, features, labels):
        """Filter outliers using Local Outlier Factor."""
        if not self.use_outlier_filter:
            return features, labels

        lof = LocalOutlierFactor(n_neighbors=self.lof_n_neighbors, contamination=0.1)
        mask = lof.fit_predict(features) != -1

        return features[mask], labels[mask]

    def visualize_with_tsne(self, title=None):
        """Generate and save t-SNE visualization of model embeddings."""
        # Extract features from model
        output_dict = self.extract_features()

        # Sample features for visualization
        features, labels = self._sample_features(output_dict)

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            early_exaggeration=self.early_exaggeration,
            method="exact",
            random_state=42,
            init="pca",
            n_jobs=-1,  # Use all available cores
        )
        features_tsne = tsne.fit_transform(features)

        # Filter outliers if enabled
        features_filtered, labels_filtered = self._filter_outliers(
            features_tsne, labels
        )

        # Create visualization
        self._plot_tsne(features_filtered, labels_filtered, title)

    def _plot_tsne(self, features, labels, title=None):
        """Create and save the t-SNE scatter plot."""
        plt.figure(figsize=(10, 8))

        # Use a colormap with enough colors
        cmap = plt.get_cmap("tab20")
        unique_labels = np.unique(labels)

        # Plot each class
        for i, label in enumerate(unique_labels):
            idx = labels == label
            color = cmap(i / len(unique_labels))
            plt.scatter(
                features[idx, 0],
                features[idx, 1],
                c=[color],
                label=f"Class {label}",
                alpha=0.7,
                s=30,
            )

        # Formatting
        plot_title = title or getattr(self.args, "model", "t-SNE Visualization")
        plt.title(plot_title, fontsize=14, fontweight="bold")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"tsne_{timestamp}.pdf")
        plt.savefig(filename, bbox_inches="tight", dpi=300, format="pdf")
        plt.close()

        print(f"✓ Saved t-SNE visualization: {filename}")
