import os
import librosa
import numpy as np
from scipy import signal
from PIL import Image
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename="audio_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def audio_to_mel_spectrogram(audio, sr):
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=224)

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_spec_normalized = normalize_image(mel_spec_db)

    # Resize to 224x512
    mel_spec_resized = signal.resample(mel_spec_normalized, 224, axis=0)
    mel_spec_resized = signal.resample(mel_spec_resized, 512, axis=1)

    # Convert to uint8
    image = (mel_spec_resized * 255).astype(np.uint8)

    return image


def process_audio(input_file, output_dir):
    try:
        y, sr = librosa.load(input_file, sr=22050)  # Using 22050 Hz sample rate
        logging.info(f"File {input_file} loaded. Shape: {y.shape}, SR: {sr}")

        chunk_length = 10 * sr  # 20 seconds * sample rate
        chunks = [y[i : i + chunk_length] for i in range(0, len(y), chunk_length)]

        for i, chunk in enumerate(chunks):
            if len(chunk) < chunk_length:
                # Pad the last chunk if it's shorter than 20 seconds
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)))

            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(input_file))[0]}_{i+1}.png",
            )

            if not os.path.isfile(output_file):
                mel_spec_image = audio_to_mel_spectrogram(chunk, sr)
                Image.fromarray(mel_spec_image).save(output_file)
                logging.info(f"{output_file} is saved")
            else:
                logging.info(f"{output_file} already exists - skipped")

    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")


def process_all_audio(input_dir, output_dir):
    total_files = sum(len(files) for _, _, files in os.walk(input_dir))

    with tqdm(total=total_files, desc="Processing audio files") as pbar:
        for label in os.listdir(input_dir):
            label_dir = os.path.join(input_dir, label)
            if os.path.isdir(label_dir):
                output_label_dir = os.path.join(output_dir, label)
                os.makedirs(output_label_dir, exist_ok=True)

                for filename in os.listdir(label_dir):
                    if filename.endswith(".wav"):
                        input_file = os.path.join(label_dir, filename)
                        process_audio(input_file, output_label_dir)
                        pbar.update(1)


if __name__ == "__main__":
    input_dir = "data/MAD_dataset/wav_files/"
    output_dir = "data/MAD_dataset/mel_spectrogram_images/"

    os.makedirs(output_dir, exist_ok=True)
    logging.info("Processing all audio files")
    process_all_audio(input_dir, output_dir)
    logging.info("Processing completed")
