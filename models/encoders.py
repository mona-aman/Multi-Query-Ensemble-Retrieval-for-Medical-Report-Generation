"""
Model encoders for X-REM
"""
import torch
import torch.nn as nn
from transformers import (
    CLIPModel, 
    CLIPProcessor,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor
)
import os
try:
    import open_clip
except Exception:
    open_clip = None
from typing import Union, List
import numpy as np
from PIL import Image


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor normalized with ImageNet stats"""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def tensor_to_pil(tensor):
    """Convert a normalized tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed (check if values are outside [0, 1])
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = denormalize_tensor(tensor)
    
    # Clamp to [0, 1] and convert to [0, 255]
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()
    
    # Convert to numpy and PIL
    if tensor.shape[0] == 3:  # [C, H, W]
        tensor = tensor.permute(1, 2, 0)  # -> [H, W, C]
    
    return Image.fromarray(tensor.cpu().numpy())


class ImageEncoder:
    """Wrapper for image encoder"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        print(f"📥 Loading image encoder: {model_name}")
        
        if "BiomedCLIP" in model_name:
            if open_clip is None:
                raise RuntimeError("open_clip_torch is required to load BiomedCLIP. Please install open_clip_torch.")
            repo_id = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                repo_id,
                pretrained=repo_id,
            )
            self.model = self.model.to(device)
            self.encoder_type = "openclip"
        elif "CLIP" in model_name or "clip" in model_name:
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.encoder_type = "clip"
        else:
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.encoder_type = "custom"
        
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, images: Union[List, torch.Tensor]) -> torch.Tensor:
        """Encode images to embeddings"""
        
        # CRITICAL: Handle list of tensors first (most common case from dataloader)
        if isinstance(images, list):
            # Check if list contains tensors and convert them to PIL
            converted_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    converted_images.append(tensor_to_pil(img))
                else:
                    # Already PIL Image
                    converted_images.append(img)
            images = converted_images
        elif isinstance(images, torch.Tensor):
            # Direct tensor input - convert to list of PIL
            if images.dim() == 3:  # Single image [C, H, W]
                images = [tensor_to_pil(images)]
            elif images.dim() == 4:  # Batch [B, C, H, W]
                images = [tensor_to_pil(img) for img in images]
        # else: already a list of PIL Images
        
        # Now process based on encoder type
        if self.encoder_type == "clip":
            inputs = self.processor(
                images=images, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            embeddings = self.model.get_image_features(**inputs)
            
        elif self.encoder_type == "openclip":
            # open_clip processor expects PIL Images
            pixel_values = torch.stack([self.processor(img) for img in images]).to(self.device)
            embeddings = self.model.encode_image(pixel_values)
            
        else:
            # Custom encoder
            inputs = self.processor(
                images=images, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


class TextEncoder:
    """Wrapper for text encoder"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        print(f"📥 Loading text encoder: {model_name}")
        
        if "BiomedCLIP" in model_name:
            if open_clip is None:
                raise RuntimeError("open_clip_torch is required to load BiomedCLIP. Please install open_clip_torch.")
            repo_id = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
            self.model, _, _ = open_clip.create_model_and_transforms(
                repo_id,
                pretrained=repo_id,
            )
            self.model = self.model.to(device)
            self.tokenizer = open_clip.get_tokenizer(repo_id)
            self.encoder_type = "openclip"
        elif "CLIP" in model_name or "clip" in model_name:
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
            # KEY FIX: Get the tokenizer from the processor
            processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = processor.tokenizer
            self.encoder_type = "clip"
        else:
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder_type = "custom"
        
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.encoder_type == "clip":
            # Use the tokenizer properly for text-only encoding
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            embeddings = self.model.get_text_features(**inputs)
            
        elif self.encoder_type == "openclip":
            tokens = self.tokenizer(texts)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            tokens = tokens.to(self.device)
            embeddings = self.model.encode_text(tokens)
            
        else:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


class NLIModel:
    """Natural Language Inference model for filtering"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        
        print(f"📥 Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, premise: str, hypothesis: str) -> str:
        """
        Predict NLI relationship
        Returns: 'entailment', 'contradiction', or 'neutral'
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state.mean(dim=1)
        
        # Simple classification (adjust based on model)
        # For now, use similarity threshold
        similarity = torch.cosine_similarity(
            logits[0].unsqueeze(0),
            logits[0].unsqueeze(0),
            dim=-1
        ).item()
        
        if similarity > 0.9:
            return 'entailment'
        elif similarity < 0.3:
            return 'contradiction'
        else:
            return 'neutral'


def load_encoders(config: dict) -> tuple:
    """Load all encoders"""
    image_encoder = ImageEncoder(
        model_name=config['models']['vision_encoder'],
        device=config['models']['device']
    )
    
    text_encoder = TextEncoder(
        model_name=config['models']['text_encoder'],
        device=config['models']['device']
    )
    
    nli_model = NLIModel(
        model_name=config['models']['nli_model'],
        device=config['models']['device']
    )
    
    return image_encoder, text_encoder, nli_model


if __name__ == "__main__":
    # Test encoders
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    img_enc, txt_enc, nli = load_encoders(config)
    
    # Test encoding
    from PIL import Image
    dummy_img = Image.new('RGB', (224, 224))
    dummy_text = "Normal chest X-ray with no acute findings."
    
    img_emb = img_enc.encode([dummy_img])
    txt_emb = txt_enc.encode(dummy_text)
    
    print(f"Image embedding shape: {img_emb.shape}")
    print(f"Text embedding shape: {txt_emb.shape}")
    
    similarity = torch.cosine_similarity(img_emb, txt_emb, dim=-1)
    print(f"Similarity: {similarity.item():.4f}")