import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
# from torch_cluster import knn

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # Shared MLP (implemented as Conv1D with kernel size 1)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Fully connected layers for global feature
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(256, num_classes)
        
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Transpose to (B, 3, N) for Conv1D
        points = points.transpose(2, 1)

        # Shared MLP
        points = self.mlp1(points)  # (B, 64, N)
        points = self.mlp2(points)  # (B, 128, N)
        points = self.mlp3(points)  # (B, 1024, N)

        # Symmetric function: max pooling
        points = torch.max(points, 2)[0]  # (B, 1024)

        # Fully connected layers
        points = F.relu(self.bn1(self.fc1(points)))
        points = self.dropout1(points)

        points = F.relu(self.bn2(self.fc2(points)))
        points = self.dropout2(points)

        points = self.fc3(points)
        #print("Shape of pred before softmax:", points.shape)
        # points = F.softmax(self.fc3(points), dim=1)  # (B, num_classes) 
        #print("Shape of pred after softmax:", points.shape)
        #points = torch.argmax(points, dim=1)    # Predicting the class
        return points



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1), 
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 256, 1), 
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, num_seg_classes, 1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _ = points.size()

        x = points.transpose(1, 2)

        l_feat = self.mlp1(points.transpose(1, 2))  # (B, 64, N)
        x = self.mlp2(l_feat)  # (B, 128, N)
        x = self.mlp3(x)  # (B, 1024, N)

        global_feat = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, N)  # (B, 1024, N)

        # Concatenate local (64) + global (1024)
        x = torch.cat([l_feat, global_feat], dim=1)  # (B, 1088, N)
        
        x = self.seg_head(x)

        x = x.transpose(1, 2)
        # print("output shape: ", x.shape)
        return x
    

# Classes for Q4 - Locality

class local_cls_model(nn.Module):
    def __init__(self, num_classes=3, k=8):
        super(local_cls_model, self).__init__()
        self.k = k

        # Shared MLP for feature extraction
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Fully connected layers for global feature
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        B, N, _ = points.shape

        # coords = points  # [B, N, 3]
        idx = (knn_points(points, points, K=self.k, return_nn=False)).idx
        points = points.transpose(2, 1)  # [B, 3, N]

        points = self.mlp1(points)
        points = self.mlp2(points)
        points = self.mlp3(points)  # [B, 1024, N]

        # Get k-NN indices using pytorch3d
        # idx = (knn_points(coords, coords, K=self.k, return_nn=False)).idx # (B, N, k)

        # Gather features from k neighbors
        # We need to use batched indexing
        idx_expanded = idx.unsqueeze(1).expand(-1, 1024, -1, -1)  # [B, 1024, N, k]
        points_expanded = points.unsqueeze(3).expand(-1, -1, -1, self.k)  # [B, 1024, N, k]
        local_features = torch.gather(points_expanded, 2, idx_expanded)  # [B, 1024, N, k]

        local_features = torch.max(local_features, dim=3)[0]  # [B, 1024, N]
        global_feature = torch.max(local_features, dim=2)[0]  # [B, 1024]

        points = F.relu(self.bn1(self.fc1(global_feature)))
        points = self.dropout1(points)

        points = F.relu(self.bn2(self.fc2(points)))
        points = self.dropout2(points)

        points = self.fc3(points)
        return points
    
class local_seg_model(nn.Module):
    def __init__(self, num_seg_classes=6, k=8):
        super(local_seg_model, self).__init__()
        self.k = k  # Number of nearest neighbors to consider for locality

        # MLP layers (same as before)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # self.mlp2 = nn.Sequential(
        #     nn.Conv1d(64, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        # )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1), 
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 256, 1), 
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, num_seg_classes, 1),
        )

    def forward(self, points):
        B, N, _ = points.size()

        # Compute k-NN indices
        # knn = knn_points(points, points, K=self.k, return_nn=False)
        idx = knn_points(points, points, K=self.k, return_nn=False).idx  # (B, N, k)

        # Transpose for MLP layers
        points = points.transpose(1, 2)  # (B, 3, N)

        # Feature extraction
        l_feat = self.mlp1(points)  # (B, 64, N)
        x = self.mlp3(l_feat)             # (B, 128, N)
        # x = self.mlp3(x)                  # (B, 512, N)
    
        # Use knn idx to gather features from x
        # x: (B, 512, N) → (B, N, 512)
        # x_t = x.transpose(1, 2)  # (B, N, 512)

        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 512)  # (B, N, k, 512)
        neighbor_feats = torch.gather(x.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, 512), dim=1, index=idx_expanded)  # (B, N, k, 512)
        local_features = neighbor_feats.max(dim=2)[0]  # (B, N, 512)

        local_features = local_features.transpose(1, 2)

        # Global feature
        global_feat = x.max(dim=2, keepdim=True)[0].repeat(1, 1, N)  # (B, 512, N)
        x = torch.cat([l_feat, global_feat], dim=1)
        # Concatenate local and global
        x = torch.cat([local_features, x], dim=1)  # (B, 512+64+512, N)

        # Segmentation head
        x = self.seg_head(x)  # (B, num_seg_classes, N)
        return x.transpose(1, 2)  # (B, N, num_seg_classes)
