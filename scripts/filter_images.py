import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.measure import label, regionprops

class_labels = {
    0: "Communication",
    1: "Gunshot",
    2: "Footsteps",
    3: "Shelling",
    4: "Vehicle",
    5: "Helicopter",
    6: "Fighter",
}


def is_continuous(image, threshold=0.9, min_area_ratio=0.5):
    # Convert to binary
    binary = image > np.mean(image)

    # Label connected components
    labeled = label(binary)

    # Get properties of labeled regions
    regions = regionprops(labeled)

    if not regions:
        return False

    # Sort regions by area
    regions.sort(key=lambda x: x.area, reverse=True)

    # Calculate the ratio of the largest region to the total area
    largest_region_ratio = regions[0].area / (image.shape[0] * image.shape[1])

    # Check if the largest region covers at least min_area_ratio of the image
    if largest_region_ratio < min_area_ratio:
        return False

    # Calculate the percentage of non-zero pixels in the largest region
    region_mask = labeled == regions[0].label
    non_zero_ratio = np.sum(image[region_mask] > 0) / regions[0].area

    return non_zero_ratio >= threshold


def has_significant_padding(image, padding_threshold=0.1):
    # Check the last 10% of columns
    last_tenth = image[:, int(0.9 * image.shape[1]) :]

    # If more than padding_threshold of this region is zero, consider it padded
    return np.mean(last_tenth == 0) > padding_threshold


def filter_spectrograms(
    input_dir,
    output_dir,
    continuity_threshold=0.9,
    min_area_ratio=0.5,
    padding_threshold=0.1,
):
    os.makedirs(output_dir, exist_ok=True)

    total_images = sum(
        len(files)
        for _, _, files in os.walk(input_dir)
        if any(f.endswith(".png") for f in files)
    )

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for class_num, class_name in class_labels.items():
            input_class_dir = os.path.join(input_dir, str(class_num))
            output_class_dir = os.path.join(output_dir, str(class_num))
            os.makedirs(output_class_dir, exist_ok=True)

            for filename in os.listdir(input_class_dir):
                if filename.endswith(".png"):
                    input_path = os.path.join(input_class_dir, filename)
                    output_path = os.path.join(output_class_dir, filename)

                    with Image.open(input_path) as img:
                        image_array = np.array(img)

                    if is_continuous(
                        image_array, continuity_threshold, min_area_ratio
                    ) and not has_significant_padding(
                        image_array,
                        padding_threshold,
                    ):
                        shutil.copy2(input_path, output_path)

                pbar.update(1)

    print("Filtering completed. New dataset created.")


# Usage
input_dir = "data/MAD_dataset/mel_spectrogram_images/"
output_dir = "data/MAD_dataset/filtered_mel_spectrograms/"
filter_spectrograms(input_dir, output_dir)
