import os
import cv2
import torch
from torch.utils.data import Dataset

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        # 获取当前文件所在目录作为基准路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 读取文件列表（兼容多种编码）
        lines = []
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'gbk', 'latin-1']:
            try:
                with open(list_file, 'r', encoding=encoding) as file:
                    lines = [line.strip() for line in file if line.strip()]
                break
            except UnicodeDecodeError:
                continue
        
        # 关键修复：将所有路径转为绝对路径
        self.image_filenames = []
        for line in lines:
            # 使用 os.path.normpath 规范化路径，自动处理 ./ 和 \ 
            normalized = os.path.normpath(line)
            
            # 如果是相对路径，拼接成绝对路径
            if not os.path.isabs(normalized):
                abs_path = os.path.join(self.base_dir, normalized)
            else:
                abs_path = normalized
            
            self.image_filenames.append(abs_path)
        
        print(f"Loaded {len(self.image_filenames)} images from {list_file}")
        
        # 验证第一张图片是否可读
        if len(self.image_filenames) > 0:
            test_img = cv2.imread(self.image_filenames[0])
            if test_img is None:
                print(f"WARNING: First image unreadable: {self.image_filenames[0]}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        
        # 读取图片
        img_color_semantic = cv2.imread(image_path)
        
        # 如果读取失败，抛出明确的错误信息
        if img_color_semantic is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # 分割左右两半（Pix2Pix 标准格式）
        w = img_color_semantic.shape[1] // 2
        img_semantic = img_color_semantic[:, :w, :]  # 左半：输入语义图
        img_rgb = img_color_semantic[:, w:, :]       # 右半：目标真实图
        
        # 转换为 tensor 并归一化到 [-1, 1]
        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        img_semantic = torch.from_numpy(img_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        
        return img_rgb, img_semantic