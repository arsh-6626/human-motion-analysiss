#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from collections import Counter

class LSTMModel(nn.Module):
    """
    LSTM-based classifier that takes a sequence of pose keypoints and outputs a class prediction.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(last_time_step)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def parse_arguments():
    parser = ArgumentParser(description="Run inference on saved pose data using a trained LSTM model")
    parser.add_argument('--input-file', type=str, default='pose_data.npy',
                        help='Path to the saved pose sequences (.npy file)')
    parser.add_argument('--model-path', type=str, default='lstm_pose_model.pth',
                        help='Path to the saved LSTM model weights')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (e.g., cuda:0 or cpu)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load the saved pose sequences
    sequences = np.load(args.input_file)
    if sequences.size == 0:
        print("No pose sequences found in the input file.")
        return
    print(f"Loaded {len(sequences)} pose sequence(s) from {args.input_file}")
    
    # Define the LSTM model with the same parameters as during training
    input_size = 34  # 17 keypoints * 2 coordinates per frame
    hidden_size = 64
    num_classes = 3  # e.g., "absent", "twitching", "walking"
    model = LSTMModel(input_size, hidden_size, num_classes).to(args.device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    # Convert the pose sequences to a tensor for inference
    sequences_tensor = torch.FloatTensor(sequences).to(args.device)
    
    with torch.no_grad():
        outputs = model(sequences_tensor)
        # Get predicted class indices (e.g., 0, 1, or 2)
        _, predictions = torch.max(outputs, dim=1)
        predictions = predictions.cpu().numpy()
    
    # Map numeric predictions to class labels
    class_map = {0: "absent", 1: "twitching", 2: "walking"}
    
    # Print the prediction for each sequence
    print("\nSequence predictions:")
    for idx, pred in enumerate(predictions):
        print(f"  Sequence {idx+1}: {class_map[pred]}")
    
    # Compute an overall prediction via majority vote
    overall_pred = Counter(predictions).most_common(1)[0][0]
    print(f"\nOverall prediction for the video: {class_map[overall_pred]}")

if __name__ == '__main__':
    main()
