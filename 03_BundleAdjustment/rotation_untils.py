import torch

def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    将 Euler 角转换为旋转矩阵
    euler_angles: (*, 3) 张量，单位为弧度
    convention: 旋转顺序，默认 "XYZ"
    returns: (*, 3, 3) 旋转矩阵
    """
    x, y, z = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)
    
    # 绕 X 轴旋转
    R_x = torch.zeros(*euler_angles.shape[:-1], 3, 3, device=euler_angles.device)
    R_x[..., 0, 0] = 1
    R_x[..., 1, 1] = cos_x
    R_x[..., 1, 2] = -sin_x
    R_x[..., 2, 1] = sin_x
    R_x[..., 2, 2] = cos_x
    
    # 绕 Y 轴旋转
    R_y = torch.zeros_like(R_x)
    R_y[..., 0, 0] = cos_y
    R_y[..., 0, 2] = sin_y
    R_y[..., 1, 1] = 1
    R_y[..., 2, 0] = -sin_y
    R_y[..., 2, 2] = cos_y
    
    # 绕 Z 轴旋转
    R_z = torch.zeros_like(R_x)
    R_z[..., 0, 0] = cos_z
    R_z[..., 0, 1] = -sin_z
    R_z[..., 1, 0] = sin_z
    R_z[..., 1, 1] = cos_z
    R_z[..., 2, 2] = 1
    
    # R = R_z @ R_y @ R_x
    if convention == "XYZ":
        return R_z @ R_y @ R_x
    else:
        raise ValueError(f"Unsupported convention: {convention}")