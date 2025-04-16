from config import TrainingConfig
from train import train
from utils.logger import logger

def main():
    config = TrainingConfig()
    
    logger.setup(config)
    
    logger.info("-"*80)
    logger.info("Training Configuration:")
    logger.info("-"*80)
    for key, value in vars(config).items():
        logger.info(f"{key:25s}: {value}")
    logger.info("-"*80)
    
    try:
        train(config)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("="*80)
        logger.info("Training finished")
        logger.info("="*80)

if __name__ == '__main__':
    main() 