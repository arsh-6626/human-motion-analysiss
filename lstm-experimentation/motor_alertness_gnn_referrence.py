import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from main_v8 import PoseEstimator, parse_arguments  # Provided from your project

#############################################
# 1. Dataset Definition
#############################################

class PoseDataset(Dataset):
    """
    Custom Dataset for pose tubes.
    
    Each sample includes:
      - pose_sequence: a tube of pose frames with shape (T, K, 2)
      - labels: the ground truth pose for the target frame with shape (K, 2)
      - spatial_edge_index: connectivity of joints within a frame (intra-frame edges)
      - temporal_edge_index: connectivity of joints across frames (inter-frame edges)
      - joint_types: indices (one per joint) representing the joint type
    """
    def __init__(self, pose_sequences, labels, spatial_edge_index, temporal_edge_index, joint_types):
        # Convert pose_sequences from numpy array to float tensor.
        # Shape: (N, T, K, 2) where N is the number of samples.
        self.pose_sequences = torch.tensor(pose_sequences, dtype=torch.float)
        # Convert labels (ground truth joint positions) to float tensor.
        self.labels = torch.tensor(labels, dtype=torch.float)
        # Convert spatial graph connectivity (numpy array) to long tensor.
        self.spatial_edge_index = torch.tensor(spatial_edge_index, dtype=torch.long)
        # Convert temporal graph connectivity (numpy array) to long tensor.
        self.temporal_edge_index = torch.tensor(temporal_edge_index, dtype=torch.long)
        # Convert joint types for each sample (shape: (N, K)) to long tensor.
        self.joint_types = torch.tensor(joint_types, dtype=torch.long)
    
    def __len__(self):
        # Return the total number of samples.
        return len(self.pose_sequences)
    
    def __getitem__(self, idx):
        # For the given index, return a tuple containing:
        # pose sequence, label, spatial and temporal edge connectivity, and joint types.
        return (self.pose_sequences[idx],
                self.labels[idx],
                self.spatial_edge_index,
                self.temporal_edge_index,
                self.joint_types[idx])

#############################################
# 2. Attention-Based Message Passing Layer
#############################################

class AttentionGNNLayer(MessagePassing):
    """
    A message passing layer that implements self-attention.
    
    It computes query, key, and value vectors from node features,
    aggregates messages from neighbors using attention, and then
    updates node features via an MLP with a residual connection.
    """
    def __init__(self, in_channels, out_channels):
        super(AttentionGNNLayer, self).__init__(aggr='add')  # Use summation to aggregate messages.
        # Linear layer to compute query vector from input features.
        self.lin_q = nn.Linear(in_channels, out_channels, bias=False)
        # Linear layer to compute key vector.
        self.lin_k = nn.Linear(in_channels, out_channels, bias=False)
        # Linear layer to compute value vector.
        self.lin_v = nn.Linear(in_channels, out_channels, bias=False)
        # MLP that combines the original features with aggregated messages.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),  # Combine original and aggregated features.
            nn.ReLU(),  # Activation function.
            nn.Linear(out_channels, out_channels)  # Final linear transformation.
        )
    
    def forward(self, x, edge_index):
        # Compute query, key, and value vectors.
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)
        # Propagate messages along edges (edge_index provided).
        agg = self.propagate(edge_index, x=x, q=q, k=k, v=v)
        # Concatenate original node features with aggregated messages,
        # pass through the MLP, and add a residual connection.
        out = self.mlp(torch.cat([x, agg], dim=-1)) + x
        return out

    def message(self, q_i, k_j, v_j):
        """
        For each edge from source node j to target node i:
          - q_i: query vector of target node i.
          - k_j: key vector of source node j.
          - v_j: value vector of source node j.
        Compute an attention weight and return the weighted value.
        """
        # Compute dot product between query and key, then apply leaky ReLU.
        alpha = F.leaky_relu((q_i * k_j).sum(dim=-1, keepdim=True))
        # Multiply the value vector by the computed attention weight.
        return v_j * alpha

#############################################
# 3. Pose Dynamics GNN Model (Prediction)
#############################################

class PoseDynamicsGNN(nn.Module):
    def __init__(self, input_dim=2, visual_dim=2, hidden_dim=64, output_dim=2, num_joints=17, num_layers=4):
        """
        PoseDynamicsGNN predicts the pose in the current frame based on a tube of historical poses.
        
        Args:
          input_dim: Dimension of the 2D joint positions (typically 2 for (x,y)).
          visual_dim: Dimension of a dummy visual feature (can be replaced with CNN features).
          hidden_dim: Hidden dimension used in the encoders and GNN layers.
          output_dim: Dimension of the predicted output (2 for (x,y) coordinates).
          num_joints: Number of joints per pose.
          num_layers: Total number of message passing layers (should be even to alternate spatial and temporal).
        """
        super(PoseDynamicsGNN, self).__init__()
        self.num_joints = num_joints

        # Encoder for dummy visual features.
        self.mlp_vis = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.ReLU())
        # Encoder for 2D position features.
        self.mlp_pos = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # Encoder for joint type features: maps joint index to a learned vector.
        self.mlp_type = nn.Embedding(num_joints, hidden_dim)
        # The final node feature dimension after concatenating the three cues.
        self.node_in_dim = hidden_dim * 3

        # Split message passing into alternating spatial and temporal layers.
        n_layers = num_layers // 2  # Number of spatial and temporal layers.
        # Create a list of spatial message passing layers.
        self.spatial_layers = nn.ModuleList([
            AttentionGNNLayer(self.node_in_dim, self.node_in_dim) for _ in range(n_layers)
        ])
        # Create a list of temporal message passing layers.
        self.temporal_layers = nn.ModuleList([
            AttentionGNNLayer(self.node_in_dim, self.node_in_dim) for _ in range(n_layers)
        ])
        
        # Final prediction head that regresses node features to (x,y) coordinates.
        self.mlp_pred = nn.Sequential(
            nn.Linear(self.node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, pose_sequence, spatial_edge_index, temporal_edge_index, joint_types):
        """
        Forward pass for predicting the current pose.
        
        Args:
          pose_sequence: Tensor of shape (batch, T, num_joints, 2) containing T historical frames.
          spatial_edge_index: Tensor of shape (2, E_spatial) for intra-frame (spatial) edges.
          temporal_edge_index: Tensor of shape (2, E_temporal) for inter-frame (temporal) edges.
          joint_types: Tensor of shape (batch, num_joints) with joint type indices.
        
        Returns:
          pred: Tensor of shape (batch, num_joints, 2) with predicted joint positions.
        """
        batch_size, T, K, _ = pose_sequence.shape  # Extract batch size, number of frames, and joints.
        
        # Flatten pose_sequence over batch, time, and joints: shape becomes (B*T*K, 2).
        x_pos = pose_sequence.view(batch_size * T * K, -1)
        # Use the same positions as dummy visual features; replace if real features are available.
        x_vis = pose_sequence.view(batch_size * T * K, -1)
        # Repeat joint types for T frames. Original shape (B, K) becomes (B, T, K) then flattened to (B*T*K).
        joint_types_rep = joint_types.unsqueeze(1).repeat(1, T, 1).view(batch_size * T * K)
        
        # Encode the visual cues.
        feat_vis = self.mlp_vis(x_vis)  # (B*T*K, hidden_dim)
        # Encode the positional information.
        feat_pos = self.mlp_pos(x_pos)  # (B*T*K, hidden_dim)
        # Encode joint types using the embedding layer.
        feat_type = self.mlp_type(joint_types_rep)  # (B*T*K, hidden_dim)
        
        # Concatenate all three features along the feature dimension.
        node_features = torch.cat([feat_vis, feat_pos, feat_type], dim=-1)  # (B*T*K, 3*hidden_dim)
        
        # Alternate between spatial and temporal message passing layers.
        n_layers = len(self.spatial_layers)
        for i in range(n_layers):
            # Update node features using spatial (intra-frame) connectivity.
            node_features = self.spatial_layers[i](node_features, spatial_edge_index)
            # Then update node features using temporal (inter-frame) connectivity.
            node_features = self.temporal_layers[i](node_features, temporal_edge_index)
        
        # Reshape node features back to (batch, T, num_joints, feature_dim).
        node_features = node_features.view(batch_size, T, K, -1)
        # Select features corresponding to the last frame (target frame).
        last_frame_features = node_features[:, -1, :, :]
        
        # Predict joint positions from the last frame's features.
        pred = self.mlp_pred(last_frame_features)  # (batch, num_joints, 2)
        return pred

#############################################
# 4. Graph Construction Helpers
#############################################

def build_spatial_edge_index(T, num_joints):
    """
    Build spatial (intra-frame) edges for a tube of T frames.
    For each frame, connect every joint with every other joint (excluding self-loops).
    
    Returns:
      spatial_edge_index: numpy array of shape (2, E_spatial)
    """
    spatial_edges = []
    for t in range(T):
        base = t * num_joints  # Offset for joints in frame t.
        for i in range(num_joints):
            for j in range(num_joints):
                if i != j:
                    spatial_edges.append((base + i, base + j))
    spatial_edge_index = np.array(spatial_edges).T  # Transpose to shape (2, E_spatial)
    return spatial_edge_index

def build_temporal_edge_index(T, num_joints):
    """
    Build temporal (inter-frame) edges for a tube of T frames.
    For each joint in frame t, connect to the same joint in frame t+1 (bidirectionally).
    
    Returns:
      temporal_edge_index: numpy array of shape (2, E_temporal)
    """
    temporal_edges = []
    for t in range(T - 1):
        base_from = t * num_joints
        base_to = (t + 1) * num_joints
        for i in range(num_joints):
            # Connect joint i in frame t to joint i in frame t+1.
            temporal_edges.append((base_from + i, base_to + i))
            # Also add the reverse connection (bidirectional).
            temporal_edges.append((base_to + i, base_from + i))
    temporal_edge_index = np.array(temporal_edges).T  # Transpose to shape (2, E_temporal)
    return temporal_edge_index

#############################################
# 5. Training Function
#############################################

def train(model, dataloader, criterion, optimizer, epochs=50, device='cpu'):
    # Move model to the specified device (CPU or GPU).
    model.to(device)
    # Set the model to training mode.
    model.train()
    # Loop over the specified number of epochs.
    for epoch in range(epochs):
        total_loss = 0.0  # Initialize running loss for this epoch.
        # Iterate over batches from the DataLoader.
        for poses, labels, spatial_edge_index, temporal_edge_index, joint_types in dataloader:
            # Move batch tensors to the designated device.
            poses = poses.to(device)              # (B, T, K, 2)
            labels = labels.to(device)            # (B, K, 2)
            spatial_edge_index = spatial_edge_index.to(device)
            temporal_edge_index = temporal_edge_index.to(device)
            joint_types = joint_types.to(device)
            
            # Zero the gradients in the optimizer.
            optimizer.zero_grad()
            # Forward pass: compute the predicted poses.
            output = model(poses, spatial_edge_index, temporal_edge_index, joint_types)  # (B, K, 2)
            # Compute the mean squared error loss between prediction and ground truth.
            loss = criterion(output, labels)
            # Backward pass: compute gradients.
            loss.backward()
            # Update model parameters.
            optimizer.step()
            # Accumulate loss value.
            total_loss += loss.item()
        
        # Print the average loss for this epoch.
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

#############################################
# 6. Main Script: Data Preparation, Model, and Training
#############################################

if __name__ == '__main__':
    # Specify the path to your video file.
    video_path = "your_video.mp4"   # Replace with your video file path
    num_joints = 17                 # Total number of joints per pose

    # -----------------------------------------------------------------
    # Step 1: Extract keypoints from the video using the provided estimator.
    # -----------------------------------------------------------------
    print("Extracting keypoints from video...")
    # Use the PoseEstimator (from main_v8) to process the video.
    keypoints = np.array(PoseEstimator(parse_arguments()).process_video())
    # keypoints shape: (F, K, 3) where F = total frames, K = joints, 3 = (x, y, confidence)
    num_frames_total = keypoints.shape[0]
    
    # -----------------------------------------------------------------
    # Step 2: Create tubes (overlapping sequences) from keypoints.
    # For each tube, use T frames (history) and the last frame as the label.
    # -----------------------------------------------------------------
    T = 4  # Number of history frames per tube
    samples = []
    labels = []
    for i in range(num_frames_total - T):
        # Create a tube of T consecutive frames.
        tube = keypoints[i: i+T]  # shape: (T, K, 3)
        # Use only (x, y) coordinates for each joint.
        samples.append(tube[:, :, :2])
        # The label is the (x, y) of joints in the last frame of the tube.
        labels.append(keypoints[i+T-1, :, :2])
    
    # Convert the lists to numpy arrays.
    samples = np.array(samples)  # shape: (N, T, K, 2)
    labels = np.array(labels)    # shape: (N, K, 2)
    
    # -----------------------------------------------------------------
    # Step 3: Build graph connectivity indices (spatial and temporal).
    # -----------------------------------------------------------------
    spatial_edge_index = build_spatial_edge_index(T, num_joints)    # (2, E_spatial)
    temporal_edge_index = build_temporal_edge_index(T, num_joints)    # (2, E_temporal)
    
    # -----------------------------------------------------------------
    # Step 4: Prepare joint type information.
    # Assume joint ordering is fixed; create an array with indices [0, 1, ..., num_joints-1]
    # and repeat it for every sample.
    # -----------------------------------------------------------------
    joint_types = np.tile(np.arange(num_joints), (samples.shape[0], 1))  # shape: (N, K)
    
    # -----------------------------------------------------------------
    # Step 5: Create the dataset and DataLoader.
    # -----------------------------------------------------------------
    dataset = PoseDataset(samples, labels, spatial_edge_index, temporal_edge_index, joint_types)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # -----------------------------------------------------------------
    # Step 6: Instantiate the model, optimizer, and loss function.
    # -----------------------------------------------------------------
    model = PoseDynamicsGNN(input_dim=2, visual_dim=2, hidden_dim=64, output_dim=2,
                            num_joints=num_joints, num_layers=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # -----------------------------------------------------------------
    # Step 7: Train the model.
    # -----------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, dataloader, criterion, optimizer, epochs=50, device=device)
