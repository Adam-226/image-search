"""
构建图像索引的命令行工具
"""

import argparse
from image_retrieval import ImageRetrieval


def main():
    parser = argparse.ArgumentParser(description="构建图像检索索引")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="images",
        help="包含图像的文件夹路径（默认: images）"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="image_index.pkl",
        help="索引文件保存路径（默认: image_index.pkl）"
    )
    
    args = parser.parse_args()
    
    # 创建检索系统并构建索引
    retrieval = ImageRetrieval()
    retrieval.build_index(args.image_folder, args.index_path)
    
    print("\n索引构建完成！")


if __name__ == "__main__":
    main()


