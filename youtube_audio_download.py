import pandas as pd
import yt_dlp
import os


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def download_audio(url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False


if __name__ == "__main__":
    # Read csv file
    df = pd.read_csv("./mad_dataset_annotation.csv")

    # Base output path
    base_output_path = "./data/MAD_dataset/wav_files"
    create_folder(base_output_path)

    # Group by Video_num to process each video only once
    grouped = df.groupby("Video_num")

    for video_num, group in grouped:
        # Get the first row for this video (assuming all rows for a video have the same URL and title)
        row = group.iloc[0]

        # Create a folder for this label
        label = row["Label"]
        label_folder = os.path.join(base_output_path, str(label))
        create_folder(label_folder)

        # Format the video number
        formatted_num = f"{int(video_num):03d}"

        # Set the output path for this video
        output_path = os.path.join(label_folder, f"{formatted_num}")

        # Download the audio
        if download_audio(row["url"], output_path):
            print(f"Successfully downloaded video {formatted_num} to {output_path}.wav")
        else:
            print(f"Skipped video {formatted_num}")

    print("Processing completed.")
