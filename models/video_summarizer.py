import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class VideoSummarizer(nn.Module):
    def __init__(self, vit_model="google/vit-base-patch16-224", bert_model="bert-base-uncased"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.fusion = nn.TransformerEncoderLayer(d_model=768*2, nhead=8)
        self.classifier = nn.Linear(768*2, 1)  # Predict frame importance

    def forward(self, frames, text_inputs):
        # Visual features (frames: [batch_size, num_frames, 3, 224, 224])
        batch_size, num_frames = frames.shape[:2]
        frame_features = []
        for i in range(num_frames):
            vit_output = self.vit(pixel_values=frames[:, i])  # [batch_size, 768]
            frame_features.append(vit_output.last_hidden_state[:, 0])
        frame_features = torch.stack(frame_features, dim=1)  # [batch_size, num_frames, 768]

        # Text features (text_inputs: [batch_size, seq_len])
        text_features = self.bert(**text_inputs).last_hidden_state[:, 0]  # [batch_size, 768]
        text_features = text_features.unsqueeze(1).repeat(1, num_frames, 1)  # [batch_size, num_frames, 768]

        # Fuse features
        combined = torch.cat([frame_features, text_features], dim=-1)  # [batch_size, num_frames, 768*2]
        combined = self.fusion(combined)
        scores = self.classifier(combined).squeeze(-1)  # [batch_size, num_frames]
        return torch.sigmoid(scores)  # Importance scores (0-1)