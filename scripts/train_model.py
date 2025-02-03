import torch
from torch.utils.data import DataLoader
from models.video_summarizer import VideoSummarizer
from transformers import ViTFeatureExtractor, BertTokenizer
from dataset import VideoSummarizationDataset  # Custom dataset (see below)

# Initialize model and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoSummarizer().to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load dataset
dataset = VideoSummarizationDataset(
    frame_dir="data/processed/frames/",
    text_data="data/raw/transcripts.txt",
    tokenizer=tokenizer,
    feature_extractor=feature_extractor
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        frames, text_inputs, labels = batch
        frames = frames.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        scores = model(frames, text_inputs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "models/saved_models/video_summarizer.pth")