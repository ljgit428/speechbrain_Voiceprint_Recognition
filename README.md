# Voice Similarity Detector

[![en](https://img.shields.io/badge/language-English-orange.svg)](README.md)
[![zh](https://img.shields.io/badge/language-中文-blue.svg)](README.zh.md)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://www.gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> For a Chinese version of this README, please see [**中文版说明**](README.zh.md).

This is a voice similarity comparison tool built with Gradio and SpeechBrain. It allows users to upload two audio files, analyzes their voiceprint features, calculates their similarity, and presents the result as an intuitive percentage.

---

## 🚀 Live Demo

You can try the application live on Hugging Face Spaces without any local installation.

**Hugging Face Space:** [**https://huggingface.co/spaces/l73jiang/Machine-Voice-Detector2**](https://huggingface.co/spaces/l73jiang/speechbrain_Voiceprint_Recognition)

---

## ✨ Features

*   **Voice Comparison**: Upload two audio files (WAV, MP3, etc.) to calculate and return their voiceprint similarity score.
*   **Result Interpretation**: Provides a clear, human-readable analysis based on the similarity score (e.g., "Highly Similar," "Moderately Similar").
*   **Multi-language Support**: Features a user-friendly interface with support for both English and Chinese, switchable at any time.
*   **Technical Transparency**: Clearly explains the core model and methodology used for the analysis.
*   **Simple Interface**: Built with Gradio for a clean, user-friendly, and straightforward experience.

---

## ⚙️ How It Works

The core workflow of this application is as follows:

1.  **Load Audio**: The application uses the `torchaudio` library to load the two audio files uploaded by the user.
2.  **Preprocessing**: To meet the model's input requirements, all audio signals are resampled to 16kHz, and any multi-channel audio is converted to a single channel (mono).
3.  **Voiceprint Extraction**: It leverages the `speechbrain/spkrec-ecapa-voxceleb` model, which is pre-trained by the **SpeechBrain** team on the **VoxCeleb** dataset. This model efficiently extracts a high-dimensional feature vector (speaker embedding) from each audio clip.
4.  **Similarity Calculation**: The cosine similarity between the two speaker embedding vectors is calculated. A value closer to 1 indicates a higher similarity between the two voiceprints.
5.  **Display Results**: The calculated similarity is converted into a percentage and displayed on the Gradio interface, along with a corresponding analysis.

---

## 🖥️ Local Setup Guide

### Prerequisites

*   Python 3.7+
*   Git

### Installation and Launch

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Please replace `your-username/your-repo-name` with your actual GitHub repository URL.)*

2.  **Install Dependencies**
    It is recommended to use a virtual environment to avoid package conflicts.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```text
    gradio
    torch
    torchaudio
    speechbrain
    scipy
    numpy
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```
    After launching, a local URL (usually `http://127.0.0.1:7860`) will be displayed in your terminal. Open this URL in your web browser to use the application.

---

## ⚠️ Disclaimer

*   This application is intended for technical demonstration and academic purposes only. The comparison results do not hold any legal value.
*   For the best results, please use clear audio clips that are longer than 1 second and have minimal background noise.

---

## 🙏 Acknowledgements

*   **SpeechBrain**: For providing the powerful `spkrec-ecapa-voxceleb` pre-trained model.
*   **Gradio**: For making it incredibly simple to build interactive interfaces for ML applications.
*   **Hugging Face**: For providing a convenient platform for hosting models and applications.
