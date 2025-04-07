import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse
import os
from typing import Optional

from src.data.dataset import VideoSummarizationDataset, collate_fn
from src.models.transformer_summarizer import TransformerSummarizer
from src.training.trainer import VideoSummarizationTrainer
from src.inference import VideoSummarizer

def setup_feature_extractor() -> torch.nn.Module:
    """Setup the pre-trained ResNet50 model for feature extraction."""
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model

def train_model(args):
    """Train the video summarization model."""
    # Setup feature extractor
    feature_extractor = setup_feature_extractor()
    
    # Create datasets
    train_dataset = VideoSummarizationDataset(
        features_dir=args.train_features_dir,
        scores_file=args.train_scores_file
    )
    
    val_dataset = VideoSummarizationDataset(
        features_dir=args.val_features_dir,
        scores_file=args.val_scores_file
    ) if args.val_features_dir else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    ) if val_dataset else None
    
    # Create model
    model = TransformerSummarizer(
        feature_dim=2048,  # ResNet50 feature dimension
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = VideoSummarizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_best=True
    )
    
    return model, history

def generate_summary(args):
    """Generate a video summary using a trained model."""
    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    
    # Create model
    model = TransformerSummarizer(
        feature_dim=2048,  # ResNet50 feature dimension
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup feature extractor
    feature_extractor = setup_feature_extractor()
    
    # Create summarizer
    summarizer = VideoSummarizer(
        model=model,
        feature_extractor=feature_extractor
    )
    
    # Generate summary
    frames = summarizer.generate_summary(
        video_path=args.video_path,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        diversity_threshold=args.diversity_threshold
    )
    
    # Save summary
    if args.output_video:
        summarizer.save_summary(frames, args.output_video)
    
    if args.output_dir:
        summarizer.save_frames(frames, args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Video Summarization System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_features_dir", required=True,
                            help="Directory containing training features")
    train_parser.add_argument("--train_scores_file", required=True,
                            help="Path to training scores file")
    train_parser.add_argument("--val_features_dir",
                            help="Directory containing validation features")
    train_parser.add_argument("--val_scores_file",
                            help="Path to validation scores file")
    train_parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size for training")
    train_parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of data loading workers")
    train_parser.add_argument("--num_epochs", type=int, default=100,
                            help="Number of training epochs")
    train_parser.add_argument("--d_model", type=int, default=512,
                            help="Transformer model dimension")
    train_parser.add_argument("--nhead", type=int, default=8,
                            help="Number of attention heads")
    train_parser.add_argument("--num_layers", type=int, default=6,
                            help="Number of transformer layers")
    train_parser.add_argument("--dim_feedforward", type=int, default=2048,
                            help="Feedforward network dimension")
    train_parser.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout rate")
    train_parser.add_argument("--checkpoint_dir", default="models",
                            help="Directory to save checkpoints")
    
    # Inference arguments
    infer_parser = subparsers.add_parser("infer", help="Generate video summary")
    infer_parser.add_argument("--checkpoint_path", required=True,
                            help="Path to model checkpoint")
    infer_parser.add_argument("--video_path", required=True,
                            help="Path to input video")
    infer_parser.add_argument("--num_frames", type=int, required=True,
                            help="Number of frames to select")
    infer_parser.add_argument("--frame_rate", type=int, default=1,
                            help="Frame sampling rate")
    infer_parser.add_argument("--diversity_threshold", type=float, default=0.5,
                            help="Minimum time distance between selected frames")
    infer_parser.add_argument("--output_video",
                            help="Path to save output video")
    infer_parser.add_argument("--output_dir",
                            help="Directory to save selected frames")
    
    # Add common model arguments to inference parser
    for arg in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]:
        infer_parser.add_argument(f"--{arg}", type=eval(train_parser.get_default(arg)))
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "infer":
        generate_summary(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 