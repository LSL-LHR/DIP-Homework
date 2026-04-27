import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rotation_untils import euler_angles_to_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BundleAdjustment(nn.Module):
    def __init__(self, n_views, n_points, image_size=(1024, 1024), 
                 focal_init=1000, dist_init=2.5):
        super().__init__()
        self.n_views = n_views
        self.n_points = n_points
        self.cx, self.cy = image_size[0]/2, image_size[1]/2
        
        # Shared focal length (all cameras)
        self.focal = nn.Parameter(torch.tensor([focal_init], dtype=torch.float32))
        
        # Camera extrinsics: Euler angles (3 params) + translation (3 params) per view
        # Initialize rotations near identity (front-facing cameras)
        self.euler_angles = nn.Parameter(torch.zeros(n_views, 3, dtype=torch.float32))
        # Initialize translations: cameras in front of object along -Z axis
        translations = torch.zeros(n_views, 3, dtype=torch.float32)
        translations[:, 2] = -dist_init  # cameras at z = -dist_init
        self.translations = nn.Parameter(translations)
        
        # 3D point positions (initialize near origin)
        point_init = torch.randn(n_points, 3, dtype=torch.float32) * 0.1
        self.points_3d = nn.Parameter(point_init)
    
    def get_rotation_matrices(self):
        """Convert Euler angles to rotation matrices"""
        return euler_angles_to_matrix(self.euler_angles, convention="XYZ")
    
    def project(self, points_3d, R, T):
        """
        Project 3D points to 2D image plane
        points_3d: (N, 3) - N个3D点
        R: (N, 3, 3) - 每个点对应的旋转矩阵
        T: (N, 3) - 每个点对应的平移向量
        returns: (u, v) each of shape (N,)
        """
        # Xc, Yc, Zc = R @ P + T
        # 用批量矩阵乘法: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
        points_cam = torch.bmm(R, points_3d.unsqueeze(-1)).squeeze(-1) + T
        
        Xc = points_cam[:, 0]
        Yc = points_cam[:, 1]
        Zc = points_cam[:, 2]
        
        # 投影到图像坐标
        u = -self.focal * Xc / Zc + self.cx
        v = self.focal * Yc / Zc + self.cy
        
        return u, v

    def forward(self, viewpoint_ids, point_ids, observed_pts):
        """
        Compute reprojection error for given correspondences
        viewpoint_ids: (N,) - 每个观测对应的视角索引
        point_ids: (N,) - 每个观测对应的3D点索引
        observed_pts: (N, 2) - 观测到的(x, y)坐标
        """
        # 获取对应的旋转矩阵: (N, 3, 3)
        R_all = self.get_rotation_matrices()  # (50, 3, 3)
        R = R_all[viewpoint_ids]  # (N, 3, 3)
        
        # 获取对应的平移向量: (N, 3)
        T = self.translations[viewpoint_ids]  # (N, 3)
        
        # 获取对应的3D点: (N, 3)
        pts_3d = self.points_3d[point_ids]  # (N, 3)
        
        # 投影
        u_pred, v_pred = self.project(pts_3d, R, T)
        
        # 计算重投影误差
        errors = torch.sqrt((u_pred - observed_pts[:, 0])**2 + 
                        (v_pred - observed_pts[:, 1])**2)
        return errors
    
def train_bundle_adjustment(points2d_data, n_views=50, n_points=20000, 
                           n_epochs=2000, batch_size=4096):
    model = BundleAdjustment(n_views, n_points).to(device)
    
    # 准备数据 - 提取所有可见观测
    observations = []
    view_keys = sorted(points2d_data.keys())
    
    for view_idx, key in enumerate(view_keys):
        obs = points2d_data[key]  # (20000, 3)
        
        for pt_idx in range(n_points):
            x, y, visibility = obs[pt_idx]
            if visibility > 0.5:  # 该点可见
                observations.append({
                    'view': view_idx,
                    'point': pt_idx,
                    'x': x,
                    'y': y
                })
    
    n_observations = len(observations)
    print(f"Total observations: {n_observations}")
    # Initialize optimizer
    optimizer = optim.Adam([
        {'params': [model.euler_angles], 'lr': 1e-4},
        {'params': [model.translations], 'lr': 1e-4},
        {'params': [model.points_3d], 'lr': 1e-3},
        {'params': [model.focal], 'lr': 1e-2}
    ])
    
    # Training loop
    losses = []
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        # Randomly sample batches
        indices = torch.randperm(n_observations)[:batch_size]
        
        batch_views = torch.tensor([observations[i]['view'] for i in indices], 
                                  dtype=torch.long, device=device)
        batch_points = torch.tensor([observations[i]['point'] for i in indices], 
                                   dtype=torch.long, device=device)
        batch_targets = torch.tensor([[observations[i]['x'], observations[i]['y']] 
                                     for i in indices], dtype=torch.float32, device=device)
        
        # Forward pass
        errors = model(batch_views, batch_points, batch_targets)
        loss = errors.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}, Focal: {model.focal.item():.2f}")
    
    return model, losses


def save_colored_obj(points_3d, colors, filename="reconstruction.obj"):
    """
    Save 3D points with colors to OBJ file
    """
    points_3d = points_3d.numpy() if torch.is_tensor(points_3d) else points_3d
    colors = colors.numpy() if torch.is_tensor(colors) else colors
    
    with open(filename, 'w') as f:
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            r, g, b = colors[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
    
    print(f"Saved {len(points_3d)} points to {filename}")


def visualize_results(model, losses, colors):
    """
    Visualize training loss and 3D reconstruction
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reprojection Error (pixels)')
    axes[0].set_title('Training Loss')
    axes[0].semilogy()
    axes[0].grid(True)
    
    # 3D point cloud (top 5000 points for visualization)
    pts = model.points_3d.detach().cpu().numpy()
    sample_idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
    pts_sample = pts[sample_idx]
    colors_sample = colors[sample_idx]
    
    axes[1].scatter(pts_sample[:, 0], pts_sample[:, 2], 
                   c=colors_sample, s=1, alpha=0.6)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('3D Point Cloud (XZ view)')
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('ba_results.png', dpi=150)
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    points2d_data = np.load("data/points2d.npz")
    colors = np.load("data/points3d_colors.npy")
    
    n_views = len(points2d_data.keys())
    n_points = len(colors)
    print(f"Data loaded: {n_views} views, {n_points} points")
    
    # Run bundle adjustment
    print("Starting Bundle Adjustment...")
    model, losses = train_bundle_adjustment(
        points2d_data, 
        n_views=n_views, 
        n_points=n_points,
        n_epochs=2000
    )
    
    # Save results
    pts_optimized = model.points_3d.detach().cpu()
    save_colored_obj(pts_optimized, colors, "data/reconstruction.obj")
    
    # Visualize results
    visualize_results(model, losses, colors)
    
    # Print final parameters
    print(f"\nFinal focal length: {model.focal.item():.2f} pixels")
    print("Camera positions:")
    for i in range(n_views):
        T = model.translations[i].detach().cpu().numpy()
        print(f"  View {i:2d}: ({T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f})")