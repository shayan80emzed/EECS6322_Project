from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    # Model configuration
    clip_model_name: str = 'ViT-B-16'
    pretrained: str = 'openai'
    
    # Dataset configuration
    dataset: str = 'imagenet'
    imagenet_root: str = '/datasets/imagenet'
    
    # Training configuration
    steps: int = 600
    warmup: int = 400
    batch_size: int = 128
    lr: float = 1e-4
    wd: float = 1e-4
    opt: str = 'adamw'
    momentum_sgd: float = 0.9
    
    # Adversarial training configuration
    attack: str = 'pgd'
    norm: str = 'huber'
    eps: float = 4.0 / 255
    eps2: float = 4.0 / 255
    iterations_adv: int = 10
    stepsize_adv: float = 1.0 / 255
    clean_weight: float = 0.0
    
    # Output configuration
    output_dir: str = 'outputs'
    save_checkpoints: bool = True
    log_freq: int = 30
    log_file: str = 'training.log'
    output_normalize: bool = False
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, self.log_file)
