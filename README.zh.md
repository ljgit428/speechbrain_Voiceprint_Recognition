# 声纹比对工具 (Voice Similarity Detector)

[![zh](https://img.shields.io/badge/language-中文-blue.svg)](README.zh.md)
[![en](https://img.shields.io/badge/language-English-orange.svg)](README.md)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://www.gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> For an English version of this README, please see [**English Version**](README.md).

这是一个基于 Gradio 和 SpeechBrain 开发的声纹比对工具。用户可以上传两个音频文件，应用会提取各自的声纹特征并计算它们的相似度，最终以百分比的形式直观展示结果。

---

## 🚀 在线体验 (Live Demo)

您可以通过下方的 Hugging Face Space 链接在线体验本应用，无需在本地安装任何环境。

**Hugging Face Space:** [**https://huggingface.co/spaces/l73jiang/Machine-Voice-Detector2**](https://huggingface.co/spaces/l73jiang/Machine-Voice-Detector2)

![应用截图](https://raw.githubusercontent.com/your-username/your-repo/main/screenshot-zh.png)
*<p align="center">（请将上方图片链接替换为您自己的项目截图）</p>*

---

## ✨ 功能特性

*   **声纹比对**: 上传两个音频文件（支持WAV、MP3等格式），计算并返回声纹相似度。
*   **结果解读**: 根据相似度得分，提供“高度相似”、“中度相似”等四种直观的分析结论。
*   **多语言支持**: 内置中文和英文两种语言，用户可以随时切换界面语言。
*   **技术透明**: 清晰说明了所使用的核心模型和技术原理。
*   **简洁界面**: 基于 Gradio 构建，界面友好，操作简单。

---

## ⚙️ 工作原理

本应用的核心工作流程如下：

1.  **加载音频**: 使用 `torchaudio` 库加载用户上传的两个音频文件。
2.  **预处理**: 将音频信号统一重采样至 16kHz，并将多声道音频转换为单声道，以符合模型输入要求。
3.  **声纹提取**: 利用 **SpeechBrain** 团队在 **VoxCeleb** 数据集上预训练的 `speechbrain/spkrec-ecapa-voxceleb` 模型来提取每个音频的声纹嵌入（Speaker Embedding）。该模型能高效地将语音片段转换为高维特征向量。
4.  **相似度计算**: 使用 Scipy 库中的 `cosine` 距离函数计算两个声纹向量之间的余弦相似度。该值越接近1，表示两个声纹越相似。
5.  **结果展示**: 将计算出的余弦相似度转换为百分比，并在 Gradio 界面上显示最终的相似度得分和对应的分析说明。

---

## 🖥️ 本地运行指南

### 环境要求

*   Python 3.7+
*   Git

### 安装与启动

1.  **克隆仓库**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *（请将 `your-username/your-repo-name` 替换为您的 GitHub 仓库地址）*

2.  **安装依赖**
    建议在虚拟环境中安装，以避免包版本冲突。
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` 文件内容如下:
    ```text
    gradio
    torch
    torchaudio
    speechbrain
    scipy
    numpy
    ```

3.  **运行应用**
    ```bash
    python app.py
    ```
    启动后，终端会显示一个本地 URL (通常是 `http://127.0.0.1:7860`)，在浏览器中打开即可使用。

---

## ⚠️ 免责声明

*   本应用仅为技术演示和学术交流目的，其比对结果不具备任何法律效力。
*   为获得最佳效果，请使用背景噪音小、语音清晰且时长超过1秒的音频片段进行比对。

---

## 🙏 致谢

*   **SpeechBrain**: 感谢其提供了强大且易于使用的 `spkrec-ecapa-voxceleb` 预训练模型。
*   **Gradio**: 感谢其让构建机器学习应用的交互界面变得如此简单。
*   **Hugging Face**: 感谢其提供了便利的模型和应用托管平台。
