import logging
import os
import sys
from datetime import datetime
import torch

class Logger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Logger._initialized:
            return
        
        self.logger = logging.getLogger('clip_aft')
        self.logger.setLevel(logging.INFO)
        Logger._initialized = True
    
    def setup(self, config):
        log_dir = os.path.join(config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        num_logs = len([f for f in os.listdir(log_dir) if f.endswith('.log')])
        log_file = os.path.join(log_dir, f'training_{num_logs+1:03d}.log')
        
        formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.handlers = []
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info("="*80)
        self.info("Starting CLIP Vision Encoder Adversarial Fine-tuning")
        self.info("="*80)
        self.info(f"Log file: {log_file}")
        self.info(f"Python version: {sys.version}")
        self.info(f"PyTorch version: {torch.__version__}")
        self.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def critical(self, message):
        self.logger.critical(message)


logger = Logger() 