"""
图像检索命令行工具
"""

import argparse
from image_retrieval import ImageRetrieval


def main():
    parser = argparse.ArgumentParser(description="根据文本描述检索图像")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="文本查询描述"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="image_index.pkl",
        help="索引文件路径（默认: image_index.pkl）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="返回前 k 个结果（默认: 5）"
    )
    
    args = parser.parse_args()
    
    # 加载索引并搜索
    retrieval = ImageRetrieval()
    retrieval.load_index(args.index_path)
    
    print(f"\n搜索查询: {args.query}\n")
    
    results = retrieval.search(args.query, args.top_k)
    
    print(f"找到 {len(results)} 个结果:\n")
    for idx, (path, score) in enumerate(results, 1):
        print(f"{idx}. {path}")
        print(f"   相似度: {score:.4f}\n")


if __name__ == "__main__":
    main()


