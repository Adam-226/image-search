# 图像检索系统

一个基于 CLIP 模型的简洁图像检索系统，可以根据文字描述从图像数据库中找到最匹配的图片。

## 功能特点

- **文本到图像检索**: 输入文字描述，AI 智能匹配最相关的图片
- **基于 CLIP 模型**: 使用 OpenAI 的 CLIP 模型，理解图像和文本的语义关系
- **简洁易用**: 提供命令行和 Web 界面两种使用方式
- **高效索引**: 预先构建图像特征索引，搜索速度快
- **灵活部署**: 支持本地运行和远程服务器部署

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 足够的内存来加载 CLIP 模型（建议 4GB+）
- （可选）GPU 用于加速索引构建

## 安装步骤

1. **克隆或下载项目**

2. **安装依赖**

```bash
pip install -r requirements.txt
```

> **国内用户注意**: 如果下载模型遇到超时，请设置 Hugging Face 镜像源：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 快速开始

### 1. 准备图像数据

创建一个 `images` 文件夹，并将你的图片放入其中：

```bash
mkdir images
# 将你的图片复制到 images 文件夹
```

支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`

### 2. 构建图像索引

首次使用前需要构建图像特征索引：

```bash
python build_index.py --image_folder images
```

这会扫描 `images` 文件夹中的所有图片，提取特征并保存到 `image_index.pkl` 文件中。

### 3. 使用方式

#### 方式一：命令行搜索

```bash
python search.py --query "a cute cat"
```

可选参数：
- `--top_k`: 返回前 k 个结果（默认 5）
- `--index_path`: 索引文件路径（默认 image_index.pkl）

示例：
```bash
python search.py --query "sunset landscape" --top_k 10
```

#### 方式二：Web 界面

**本地运行**：

```bash
python app.py
```

然后在浏览器中打开 `http://127.0.0.1:6006`

**远程服务器（AutoDL 等）**：

1. 在服务器上运行 `python app.py`
2. 在 AutoDL 控制台查看"自定义服务"链接
3. 点击链接即可在浏览器中访问

> Web 界面提供了更直观的可视化搜索体验。

## 项目结构

```
image-search/
├── image_retrieval.py    # 核心检索模块
├── build_index.py        # 构建索引脚本
├── search.py             # 命令行搜索工具
├── app.py                # Web 应用
├── templates/
│   └── index.html        # Web 界面
├── requirements.txt      # 依赖列表
├── README.md            # 说明文档
├── images/              # 图像文件夹（需要创建）
└── image_index.pkl      # 索引文件（自动生成）
```

## 使用示例

### 命令行示例

```bash
# 搜索猫的图片
python search.py --query "a cat sitting"

# 搜索风景照片
python search.py --query "beautiful sunset landscape"

# 搜索特定物品
python search.py --query "red car on the road"

# 搜索人物
python search.py --query "person walking"
```

### 索引自定义文件夹

```bash
# 索引自定义路径的图片
python build_index.py --image_folder /path/to/your/images --index_path my_index.pkl

# 使用自定义索引搜索
python search.py --query "your search query" --index_path my_index.pkl
```

### Web 界面搜索示例

在 Web 界面的搜索框中输入：
- `airplane flying in the sky`
- `cute cat looking at camera`
- `colorful flowers blooming`
- `fresh fruits on table`
- `motorcycle on road`

## 工作原理

1. **特征提取**: 使用 CLIP 模型将图像编码为高维特征向量
2. **索引构建**: 预先提取所有图像的特征并保存
3. **文本查询**: 将用户输入的文本编码为特征向量
4. **相似度计算**: 计算文本特征与所有图像特征的余弦相似度
5. **返回结果**: 返回相似度最高的前 k 张图片

## 技术栈

- **深度学习框架**: PyTorch
- **模型**: OpenAI CLIP (ViT-B/32)
- **Web 框架**: Flask
- **图像处理**: Pillow
- **数据处理**: NumPy

## 注意事项

- 首次运行会下载 CLIP 模型（约 600MB），需要网络连接
- 构建索引的时间取决于图像数量，大约每秒处理 2-10 张图片
- 如果有 GPU，会自动使用 GPU 加速，否则使用 CPU
- 图像越多，占用的内存也越多

## 使用技巧

1. **使用英文描述效果最佳**: 经过测试，英文查询的准确度明显高于中文
   - 推荐：`"a cat sitting on sofa"`
   - 次选：`"一只猫坐在沙发上"`

2. **描述要具体**: `"orange cat lying on red sofa"` 比 `"cat"` 的搜索结果更准确

3. **可以描述场景和细节**: 
   - `"airplane flying in blue sky"`
   - `"person walking in the park"`
   - `"red sports car parked"`

4. **可以描述风格**: 例如 `"watercolor landscape"`, `"black and white photo"` 等

5. **定期更新索引**: 添加新图片后，需要重新运行 `build_index.py`

## 常见问题

### Q: 模型下载超时怎么办？

A: 国内用户可能遇到 Hugging Face 连接问题，设置镜像源即可：

```bash
# 临时使用
export HF_ENDPOINT=https://hf-mirror.com
python build_index.py --image_folder images

# 或永久添加到 ~/.bashrc 或 ~/.zshrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### Q: 出现 "torch.load" 安全警告怎么办？

A: 这是 PyTorch 版本问题，代码已经配置使用更安全的 safetensors 格式，无需担心。如果仍有问题，升级 PyTorch：

```bash
pip install --upgrade torch torchvision
```

### Q: 如何在远程服务器上使用 Web 界面？

A: 代码已配置为监听所有网络接口（`0.0.0.0:6006`）：

**AutoDL 用户**：
1. 运行 `python app.py`
2. 在 AutoDL 控制台查看"自定义服务"或端口 6006 的访问链接

**其他云服务器**：
- 确保防火墙开放 6006 端口
- 通过 `http://服务器IP:6006` 访问

### Q: 为什么搜索结果不准确？

A: 几个建议：
1. **使用英文描述**（最重要！）
2. 确保图片库中有相关图片
3. 使用更具体的描述
4. 尝试不同的表述方式

### Q: 可以处理多少张图片？

A: 取决于内存大小：
- 8GB 内存：约 1000-5000 张
- 16GB 内存：约 10000+ 张
- 更大规模建议使用向量数据库（如 Faiss）

## 扩展建议

- 添加图像预处理（如调整大小、增强等）
- 支持视频帧检索
- 添加图像分类和标签功能
- 集成向量数据库（如 Faiss）以支持更大规模的图像库
- 添加用户管理和搜索历史功能

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP) - 提供强大的视觉-语言模型
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 模型加载和推理


