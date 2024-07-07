import librosa
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def audio_to_image(y, sr, method="mel"):
    if method == "mel":
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
    elif method == "cqt":
        S = np.abs(librosa.cqt(y, sr=sr))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
    elif method == "chroma":
        S = librosa.feature.chroma_stft(y=y, sr=sr)
        S_db = S  # Chroma is already normalized
    elif method == "mfcc":
        S = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        S_db = S  # MFCCs don't need to be converted to dB
    elif method == "tonnetz":
        S = librosa.feature.tonnetz(y=y, sr=sr)
        S_db = S  # Tonnetz is already normalized
    else:
        raise ValueError("Unsupported method")

    # Normalize
    img = normalize_image(S_db)

    # Resize to 224x224
    img_resized = signal.resample(img, 224, axis=0)
    img_resized = signal.resample(img_resized, 224, axis=1)

    return img_resized


def visualize_audio_representations(
    audio_file, methods=["mel", "cqt", "chroma", "mfcc", "tonnetz"]
):
    y, sr = librosa.load(audio_file, duration=10)  # Load 10 seconds of audio

    n_methods = len(methods)
    n_rows = (n_methods + 1) // 2  # Round up division
    n_cols = min(n_methods, 2)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=methods)

    for i, method in enumerate(methods):
        img = audio_to_image(y, sr, method=method)

        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(z=img, colorscale="Viridis", showscale=False), row=row, col=col
        )

    fig.update_layout(
        title_text=f"Audio Representations for {audio_file}",
        height=300 * n_rows,
        width=500 * n_cols,
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()


# Usage
audio_file = "data/MAD_dataset/wav_files/1/007.wav"
visualize_audio_representations(audio_file)
