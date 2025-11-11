"""
图像检索系统核心模块
使用 CLIP 模型进行图像和文本的特征提取
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
from typing import List, Tuple


class ImageRetrieval:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化图像检索系统
        
        Args:
            model_name: CLIP 模型名称
        """
        print("正在加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"模型已加载到 {self.device}")
        
        self.image_features = None
        self.image_paths = []
        
    def extract_image_features(self, image_path: str) -> np.ndarray:
        """
        提取单张图像的特征向量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像特征向量
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    def build_index(self, image_folder: str, index_path: str = "image_index.pkl"):
        """
        构建图像索引
        
        Args:
            image_folder: 图像文件夹路径
            index_path: 索引文件保存路径
        """
        print(f"正在扫描图像文件夹: {image_folder}")
        
        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        # 收集所有图像文件
        image_files = []
        for root, _, files in os.walk(image_folder):
            for file in files:
                if os.path.splitext(file.lower())[1] in valid_extensions:
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"警告: 在 {image_folder} 中没有找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像，开始提取特征...")
        
        # 提取所有图像的特征
        features_list = []
        self.image_paths = []
        
        for idx, img_path in enumerate(image_files):
            try:
                features = self.extract_image_features(img_path)
                features_list.append(features)
                self.image_paths.append(img_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"已处理 {idx + 1}/{len(image_files)} 张图像")
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
        
        # 合并特征
        self.image_features = np.vstack(features_list)
        
        # 保存索引
        index_data = {
            'features': self.image_features,
            'paths': self.image_paths
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"索引已保存到 {index_path}")
        print(f"总共索引了 {len(self.image_paths)} 张图像")
    
    def load_index(self, index_path: str = "image_index.pkl"):
        """
        加载图像索引
        
        Args:
            index_path: 索引文件路径
        """
        print(f"正在加载索引: {index_path}")
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.image_features = index_data['features']
        self.image_paths = index_data['paths']
        
        print(f"已加载 {len(self.image_paths)} 张图像的索引")
    
    def search(self, text_query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据文本查询检索图像
        
        Args:
            text_query: 文本查询
            top_k: 返回前 k 个最相似的图像
            
        Returns:
            (图像路径, 相似度分数) 的列表
        """
        if self.image_features is None:
            raise ValueError("请先构建或加载图像索引")
        
        # 提取文本特征
        inputs = self.processor(text=text_query, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features.cpu().numpy()
        
        # 计算相似度
        similarities = (self.image_features @ text_features.T).squeeze()
        
        # 获取 top-k 结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.image_paths[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


if __name__ == "__main__":
    # 示例使用
    retrieval = ImageRetrieval()
    
    # 构建索引（首次运行）
    if os.path.exists("images"):
        retrieval.build_index("images")
    else:
        print("请创建 'images' 文件夹并放入一些图像文件")


