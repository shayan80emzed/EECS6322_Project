import torch
import torch.nn.functional as F
import open_clip
from torchvision import transforms
from utils.logger import logger

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, vision, output_normalize=False):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

def load_clip_model(clip_model_name):
    model, image_processor = open_clip.create_model_from_pretrained(
        clip_model_name, pretrained='openai', device='cpu'
    )
    logger.info("Successfully loaded CLIP model")
    
    model.eval()
    preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
    normalizer = image_processor.transforms[-1]
    return model, preprocessor_no_norm, normalizer 