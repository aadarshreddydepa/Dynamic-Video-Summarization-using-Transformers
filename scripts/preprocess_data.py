import os
import cv2
import numpy as np
from transformers import BertTokenizer

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a specified frame rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.jpg"), frame)
        frame_count += 1
    cap.release()

def preprocess_text(text, tokenizer, max_length=128):
    """
    Tokenize text (e.g., subtitles) using BERT tokenizer.
    """
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs

if __name__ == "__main__":
    # Example usage
    video_path = "data/raw/sample_video.mp4"
    output_dir = "data/processed/frames/"
    extract_frames(video_path, output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "Sample video transcript text."
    preprocessed_text = preprocess_text(text, tokenizer)