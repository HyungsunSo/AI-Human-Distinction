"""
Inference Service Module
Handles text analysis: paragraph splitting, AI probability, Leave-One-Out importance
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from .model_loader import model_loader


class InferenceService:
    """Handles model inference and paragraph analysis"""
    
    def __init__(self):
        self.max_length = 512
    
    def split_paragraphs(self, text: str, min_length: int = 50) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > min_length]
        if len(paragraphs) < 1:
            paragraphs = [text]
        return paragraphs
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict AI probability for a list of texts
        Returns: np.ndarray of shape (n_texts, 2) with [human_prob, ai_prob]
        """
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        device = model_loader.get_device()
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        return probs
    
    def predict_single(self, text: str) -> float:
        """Predict AI probability for a single text"""
        probs = self.predict_proba([text])
        return float(probs[0, 1])  # AI probability
    
    def get_paragraph_contributions(self, paragraphs: List[str]) -> Dict:
        """
        Analyze each paragraph individually and find the most suspicious one
        """
        ai_probs = []
        
        for para in paragraphs:
            prob = self.predict_single(para)
            ai_probs.append(prob)
        
        ai_probs = np.array(ai_probs)
        top_idx = int(np.argmax(ai_probs))
        
        return {
            'ai_probs': ai_probs.tolist(),
            'top_idx': top_idx,
            'top_prob': float(ai_probs[top_idx]),
            'top_para': paragraphs[top_idx]
        }
    
    def get_paragraph_importance_loo(self, paragraphs: List[str]) -> Tuple[np.ndarray, float]:
        """
        Leave-One-Out importance: measure how much each paragraph contributes
        by removing it and measuring the score change
        """
        # Base score with all paragraphs
        full_text = '\n'.join(paragraphs)
        base_score = self.predict_single(full_text)
        
        importance = []
        
        for i in range(len(paragraphs)):
            # Remove paragraph i
            remaining = [p for j, p in enumerate(paragraphs) if j != i]
            if remaining:
                remaining_text = '\n'.join(remaining)
                loo_score = self.predict_single(remaining_text)
            else:
                loo_score = 0.5  # Neutral if only one paragraph
            
            # Importance = how much the score drops when this paragraph is removed
            importance.append(base_score - loo_score)
        
        return np.array(importance), base_score
    
    def analyze_document(self, text: str) -> Dict:
        """
        Full document analysis: split → paragraph probs → LOO importance
        """
        # Split into paragraphs
        paragraphs = self.split_paragraphs(text)
        
        # Get paragraph contributions
        contributions = self.get_paragraph_contributions(paragraphs)
        
        # Get LOO importance
        importance, base_score = self.get_paragraph_importance_loo(paragraphs)
        
        # Build paragraph info list
        paragraph_info = []
        for i, para in enumerate(paragraphs):
            paragraph_info.append({
                'index': i,
                'text': para,
                'ai_prob': contributions['ai_probs'][i],
                'importance': float(importance[i])
            })
        
        # Overall prediction
        prediction = 'AI' if base_score > 0.5 else 'Human'
        
        return {
            'prediction': prediction,
            'confidence': float(base_score),
            'paragraphs': paragraph_info,
            'top_paragraph': {
                'index': contributions['top_idx'],
                'text': contributions['top_para'],
                'ai_prob': contributions['top_prob']
            }
        }


# Singleton instance
inference_service = InferenceService()
