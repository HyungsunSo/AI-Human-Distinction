"""
Model Loader Module
Handles loading and managing BERT model checkpoints
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, List
from pathlib import Path


class ModelLoader:
    """Manages model checkpoints and loading"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = "klue/bert-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_checkpoint: Optional[str] = None
        
    def list_checkpoints(self) -> List[str]:
        """List available checkpoint files"""
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for f in self.checkpoint_dir.iterdir():
            if f.suffix in ['.pt', '.pth', '.bin']:
                checkpoints.append(f.name)
            elif f.is_dir() and (f / 'pytorch_model.bin').exists():
                checkpoints.append(f.name)
        
        return checkpoints if checkpoints else ["default"]
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load a specific checkpoint"""
        # Initialize tokenizer (always from base model)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        try:
            if checkpoint_name == "default" or not self.checkpoint_dir.exists():
                # Load base model without checkpoint
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2
                )
            else:
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                if checkpoint_path.is_dir():
                    # HuggingFace format checkpoint
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        checkpoint_path,
                        num_labels=2
                    )
                else:
                    # PyTorch state dict
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        num_labels=2
                    )
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            self.current_checkpoint = checkpoint_name
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def get_model(self) -> AutoModelForSequenceClassification:
        """Get the currently loaded model"""
        if self.model is None:
            self.load_checkpoint("default")
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
    
    def get_device(self) -> torch.device:
        """Get the current device"""
        return self.device


# Singleton instance
model_loader = ModelLoader()
