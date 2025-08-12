import gradio as gr
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine
import numpy as np
import torch

# 1. 翻译字典
translations = {
    "zh": {
        "title": "# 声纹比对逻辑模拟",
        "description": "上传两个WAV或MP3格式的音频文件，应用将分析并比较它们的声纹，输出相似度百分比。",
        "notice": "**注意：** 本应用仅为技术演示，结果不具有法律效力。为获得最佳效果，请使用清晰、无背景噪音、时长超过1秒的语音片段。",
        "language_label": "语言/Language",
        "audio1_label": "音频文件1",
        "audio2_label": "音频文件2",
        "button_text": "开始比对",
        "similarity_label": "相似度",
        "analysis_label": "结果分析",
        "tech_note": "技术说明：本应用使用 `speechbrain/spkrec-ecapa-voxceleb` 模型提取声纹特征，并通过计算声纹向量之间的余弦相似度来评估相似性。",
        "error_model_load": "声纹识别模型加载失败，请检查应用日志。",
        "error_no_file": "错误: '{file_label}' 未上传文件。",
        "error_too_short": "错误: '{file_label}' 的音频时长过短，请使用至少0.5秒以上的语音。",
        "error_processing": "无法处理 '{file_label}'。请确保它是有效的音频文件（如WAV, MP3），且内容不是静音。",
        "exp_high": "高度相似：这两个声音很可能来自同一个人。",
        "exp_medium": "中度相似：这两个声音有一些相似的特征，可能来自同一个人。",
        "exp_low": "低度相似：这两个声音有一些共同点，但差异也很明显。",
        "exp_very_low": "非常不相似：这两个声音很可能来自不同的人。"
    },
    "en": {
        "title": "# Voice Similarity Detector",
        "description": "Upload two audio files (WAV or MP3). The application will analyze and compare their voiceprints, then output a similarity percentage.",
        "notice": "**Note:** This application is for technical demonstration only and its results have no legal value. For best results, please use clear voice clips longer than 1 second without background noise.",
        "language_label": "语言/Language",
        "audio1_label": "Audio File 1",
        "audio2_label": "Audio File 2",
        "button_text": "Start Comparison",
        "similarity_label": "Similarity",
        "analysis_label": "Result Analysis",
        "tech_note": "Technical Note: This app uses the `speechbrain/spkrec-ecapa-voxceleb` model to extract voiceprint features and evaluates similarity by calculating the cosine similarity between the voiceprint vectors.",
        "error_model_load": "Failed to load the speaker recognition model. Please check the application logs.",
        "error_no_file": "Error: File not uploaded for '{file_label}'.",
        "error_too_short": "Error: The audio in '{file_label}' is too short. Please use a voice clip longer than 0.5 seconds.",
        "error_processing": "Could not process '{file_label}'. Please ensure it is a valid audio file (e.g., WAV, MP3) and is not silent.",
        "exp_high": "Highly Similar: These two voices are very likely from the same person.",
        "exp_medium": "Moderately Similar: These two voices share some similar features and might be from the same person.",
        "exp_low": "Low Similarity: These two voices have some common points, but the differences are also significant.",
        "exp_very_low": "Very Dissimilar: These two voices are very likely from different people."
    }
}
lang_map = {'中文': 'zh', 'English': 'en'}
lang_map_rev = {v: k for k, v in lang_map.items()}

# --- 模型加载 ---
def load_model():
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        return classifier
    except Exception as e:
        print(f"FATAL: 无法加载模型: {e}")
        return None
classifier = load_model()

# --- 核心功能函数 ---
def get_embedding(audio_path, file_label_key, lang):
    t = translations[lang]
    file_label = t[file_label_key]

    if not classifier:
        raise gr.Error(t["error_model_load"])
    if audio_path is None:
        raise gr.Error(t["error_no_file"].format(file_label=file_label))

    try:
        signal, fs = torchaudio.load(audio_path)
        MIN_SAMPLES = 8000
        if signal.shape[1] < MIN_SAMPLES:
            raise gr.Error(t["error_too_short"].format(file_label=file_label))
            
        if fs != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        with torch.no_grad():
            embedding = classifier.encode_batch(signal).squeeze()
        
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"处理 '{file_label}' 时出错: {e}")
        raise gr.Error(t["error_processing"].format(file_label=file_label))

def compare_voices(audio1_path, audio2_path, lang_choice):
    lang = lang_map[lang_choice]
    t = translations[lang]

    embedding1 = get_embedding(audio1_path, "audio1_label", lang)
    embedding2 = get_embedding(audio2_path, "audio2_label", lang)
    
    cosine_similarity = 1 - cosine(embedding1, embedding2)
    similarity_percentage = max(0, cosine_similarity) * 100
    
    if similarity_percentage > 85:
        explanation = t["exp_high"]
    elif similarity_percentage > 70:
        explanation = t["exp_medium"]
    elif similarity_percentage > 50:
        explanation = t["exp_low"]
    else:
        explanation = t["exp_very_low"]

    return f"{similarity_percentage:.2f}%", explanation

def update_language(lang_choice):
    """当语言选择变化时，更新所有UI组件的文本。"""
    lang = lang_map[lang_choice]
    t = translations[lang]
    # 【最终修正】: 使用正确的 gr.update() 来创建更新对象
    return (
        gr.update(value=t["title"]),
        gr.update(value=t["description"]),
        gr.update(value=t["notice"]),
        gr.update(label=t["audio1_label"]),
        gr.update(label=t["audio2_label"]),
        gr.update(value=t["button_text"]),
        gr.update(label=t["similarity_label"]),
        gr.update(label=t["analysis_label"]),
        gr.update(value=t["tech_note"])
    )

# --- Gradio 界面 ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    default_lang = "zh"
    t = translations[default_lang]

    title_md = gr.Markdown(t["title"])
    lang_selector = gr.Radio(
        list(lang_map.keys()), 
        label=t["language_label"], 
        value=lang_map_rev[default_lang]
    )
    desc_md = gr.Markdown(t["description"])
    notice_md = gr.Markdown(t["notice"])
    
    with gr.Row():
        audio1 = gr.Audio(type="filepath", label=t["audio1_label"])
        audio2 = gr.Audio(type="filepath", label=t["audio2_label"])
        
    compare_btn = gr.Button(t["button_text"], variant="primary")
    
    with gr.Column():
        similarity_output = gr.Label(label=t["similarity_label"])
        explanation_output = gr.Textbox(label=t["analysis_label"], interactive=False, lines=2)

    tech_note_md = gr.Markdown(t["tech_note"])

    lang_selector.change(
        fn=update_language,
        inputs=lang_selector,
        outputs=[
            title_md, desc_md, notice_md, audio1, audio2, 
            compare_btn, similarity_output, explanation_output, tech_note_md
        ]
    )
    
    compare_btn.click(
        fn=compare_voices,
        inputs=[audio1, audio2, lang_selector],
        outputs=[similarity_output, explanation_output]
    )

if __name__ == "__main__":
    demo.launch()