import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import albumentations as A
import torch.nn as nn
import torch.optim as optim
import config
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5), 
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.5, rotate_limit=30, p=0.5), #Зеркалит
            A.ElasticTransform(p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.Blur(blur_limit=7, always_apply=False, p=0.5),
            A.ColorJitter(brightness=.5, hue=.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        train_transform,
        val_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("../weights/last_31.tar", map_location = config.DEVICE), model)


    MAX_ACC = check_accuracy(val_loader, model, device=config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer":optimizer.state_dict()}

        # check accuracy
        acc = check_accuracy(val_loader, model, device=config.DEVICE)
        
        MAX_ACC = save_checkpoint(checkpoint, acc, MAX_ACC)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=config.DEVICE)

if __name__ == "__main__":
    main()