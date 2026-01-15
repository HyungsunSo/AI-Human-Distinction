"""
LIME Analyzer Module
Handles LIME explanations and deletion test for reliability verification
"""

import numpy as np
from typing import List, Dict, Callable
from lime.lime_text import LimeTextExplainer
from .inference import inference_service


class LimeAnalyzer:
    """LIME-based explanation and reliability testing"""
    
    def __init__(self):
        self.explainer = LimeTextExplainer(
            class_names=['Human', 'AI'],
            random_state=42
        )
    
    def _predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """Wrapper function for LIME compatibility"""
        return inference_service.predict_proba(texts)
    
    def explain_text(self, text: str, num_features: int = 15, num_samples: int = 500) -> Dict:
        """
        Generate LIME explanation for a text
        
        Args:
            text: Text to explain
            num_features: Number of top features to return
            num_samples: Number of perturbation samples
            
        Returns:
            Dict with tokens and their importance scores
        """
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba_wrapper,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract token scores
        # Positive score = AI indicator, Negative score = Human indicator
        tokens = []
        for word, score in explanation.as_list():
            tokens.append({
                'word': word,
                'score': float(score)
            })
        
        return {
            'tokens': tokens,
            'explanation': explanation  # Keep for deletion test
        }
    
    def explain_text_in_order(self, text: str, num_features: int = 30, num_samples: int = 500) -> List[Dict]:
        """
        Generate LIME explanation with tokens in original text order
        
        Args:
            text: Text to explain
            num_features: Number of top features to analyze
            num_samples: Number of perturbation samples
            
        Returns:
            List of dicts with word and score, in original text order
        """
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba_wrapper,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Build word -> score map from LIME results
        score_map = {}
        for word, score in explanation.as_list():
            score_map[word.lower()] = float(score)
        
        # Split text preserving original order
        import re
        words = re.split(r'(\s+)', text)
        
        result = []
        for word in words:
            if word.strip():  # Skip pure whitespace
                clean_word = re.sub(r'[.,!?;:\'"()[\]{}]', '', word).lower()
                score = score_map.get(clean_word, 0.0)
                result.append({
                    'word': word,
                    'score': score
                })
        
        return result
    
    def deletion_test(self, text: str, lime_result: Dict, top_k: int = 5) -> Dict:
        """
        Deletion test: Remove top AI-indicating tokens and measure probability change
        
        This verifies whether the LIME explanation is reliable:
        - If removing top tokens causes a big drop → explanation is reliable
        - If removing top tokens causes little change → explanation may be unreliable
        """
        # Original probability
        original_prob = inference_service.predict_single(text)
        
        # Get top tokens to remove (positive score = AI indicator)
        tokens = lime_result['tokens']
        top_tokens_to_remove = [
            t['word'] for t in tokens[:top_k] if t['score'] > 0
        ]
        
        # If no positive tokens, use top k anyway
        if not top_tokens_to_remove:
            top_tokens_to_remove = [t['word'] for t in tokens[:top_k]]
        
        # Remove tokens from text
        modified_text = text
        for token in top_tokens_to_remove:
            modified_text = modified_text.replace(token, ' ')
        
        # Probability after deletion
        modified_prob = inference_service.predict_single(modified_text)
        
        # Calculate drop
        drop = original_prob - modified_prob
        
        # Determine reliability
        if drop > 0.1:
            reliability = 'high'
        elif drop > 0.05:
            reliability = 'medium'
        else:
            reliability = 'low'
        
        return {
            'original_prob': float(original_prob),
            'modified_prob': float(modified_prob),
            'drop': float(drop),
            'reliability': reliability,
            'removed_tokens': top_tokens_to_remove
        }
    
    def full_analysis(self, text: str, num_features: int = 15, num_samples: int = 500) -> Dict:
        """
        Complete LIME analysis with deletion test
        """
        # LIME explanation
        lime_result = self.explain_text(text, num_features, num_samples)
        
        # Deletion test
        deletion_result = self.deletion_test(text, lime_result)
        
        return {
            'lime_result': {
                'tokens': lime_result['tokens']
            },
            'deletion_test': deletion_result
        }


# Singleton instance
lime_analyzer = LimeAnalyzer()
