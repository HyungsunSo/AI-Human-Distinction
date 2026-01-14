import streamlit as st
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_config

config = load_config("config.yaml")
max_length = config["model"]["h_param"]["max_length"]
labels = ["Human", "AI"] # 0, 1

def load_model_and_tokenizer(inference_model_dir):
    tokenizer = AutoTokenizer.from_pretrained(inference_model_dir)
    model = AutoTokenizer.from_pretrained(inference_model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def predict_prob(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        truncation = True,
        max_length = max_length,
        return_tensor = "pt",
    ).to(device)
    
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim = -1).squeeze(0)
    return probs.detach().cpu().numpy()
    
def plot_bar(probs):
    fig, ax = plt.subplots()
    ax.bar(labels, probs * 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("AI - Human Probability")
    for i, v in enumerate(probs * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha = "centor")
    return fig

def main():
    inference_model_dir = config["inference"]["inference_model_dir"]
    st.title("AI-Human 판별기 확률값 결과")
    st.set_page_config(page_title = "AI-Human Detector", layout = "centered")
    text = st.text_area("입력 텍스트", height = 200, placeholder = "텍스트 입력")
    
    tokenizer, model, device = load_model_and_tokenizer(inference_model_dir)
    
    if st.button("Start Predict"):
        if not text.strip():
            st.warning("텍스트를 입력해주세요!")
            return

        probs = predict_prob(text, tokenizer, model, device)
        p_human, p_ai = float(probs[0]), float(probs[1])
        
        pred_label = 1 if p_ai >= 0.5 else 0
        pred_name = "AI" if pred_label == 1 else "Human"
        
        st.subheader("결과")
        st.write(f"- **예측 라벨:** {pred_name}  (threshold={0.5:.2f})")
        st.write(f"- **Human 확률:** {p_human*100:.2f}%")
        st.write(f"- **AI 확률:** {p_ai*100:.2f}%")
        
        fig = plot_bar(probs)
        st.pyplot(fig)
    
if __name__ == "__main__":
    main()