import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class_labels = {
    0: "Communication",
    1: "Gunshot",
    2: "Footsteps",
    3: "Shelling",
    4: "Vehicle",
    5: "Helicopter",
    6: "Fighter",
}


def load_image(file_path):
    with Image.open(file_path) as img:
        return np.array(img)


def plot_samples(input_dir, num_samples=2):
    num_classes = len(class_labels)

    # Set up the plot style
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(16, 5 * num_classes))
    fig.suptitle("Mel Spectrograms: 2 Samples from Each Class", fontsize=20, y=1.02)

    for class_num, class_name in class_labels.items():
        class_dir = os.path.join(input_dir, str(class_num))
        image_files = [f for f in os.listdir(class_dir) if f.endswith(".png")]

        if len(image_files) < num_samples:
            print(f"Warning: Class {class_name} has fewer than {num_samples} samples.")
            continue

        samples = random.sample(image_files, num_samples)

        for j, sample in enumerate(samples):
            img = load_image(os.path.join(class_dir, sample))
            sns.heatmap(
                img,
                ax=axes[class_num, j],
                cmap="viridis",
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            axes[class_num, j].set_title(f"{class_name} - Sample {j+1}")
            axes[class_num, j].axis("off")

    plt.tight_layout()
    plt.savefig("mel_spectrogram_samples_seaborn.png", dpi=300, bbox_inches="tight")
    plt.show()


# Usage
input_dir = "data/MAD_dataset/filtered_mel_spectrograms/"
plot_samples(input_dir)
