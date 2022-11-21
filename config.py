import torch

#HYPERPARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('DEVICE', DEVICE)
BATCH_SIZE = 10
NUM_EPOCHS = 1000
NUM_WORKERS = 4
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "roboflow/image_train/"
TRAIN_MASK_DIR = "roboflow/mask_train/"
VAL_IMG_DIR = "roboflow/image_valid/"
VAL_MASK_DIR = "roboflow/mask_valid/"