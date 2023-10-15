from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
import pandas as pd
from tqdm import tqdm

from src.dataloader import CarsDataset
from src.transforms import transforms
from src.utils import get_model_predictions
from src.trainer import train_model
from src.models import get_model
from src.config import config

image_pathes = np.array(glob(f"data/techosmotr/techosmotr/train/**/*.jpeg", recursive=True))
image_classes = np.array([image_path.split("/")[-2] for image_path in image_pathes])
class_names = sorted(set(image_classes))

index2class = {i: x for i, x in enumerate(class_names)}
class2index = {x: i for i, x in index2class.items()}

skf = StratifiedKFold(n_splits=config["N_SPLITS"], shuffle=True, random_state=config["RANDOM_STATE"])
folds = {}
for i, (train_index, val_index) in enumerate(skf.split(image_pathes, image_classes)):
    folds[i] = (train_index, val_index)

for index_fold, fold in folds.items():
    print(f"Run on {index_fold} fold...")
    indexes = {"train": fold[0], "val": fold[1]}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            CarsDataset(image_pathes, image_classes, indexes[x], class2index, transforms[x]),
            batch_size=config["BATCH_SIZE"], 
            shuffle=True, 
            num_workers=config["N_WORKERS"],
        )
        for x in ['train', 'val']
    }

    class_weights = torch.tensor(config["CLASS_WEIGHTS"], dtype=torch.float).to(config["DEVICE"])

    model = get_model(config["MODEL_NAME"], config["DEVICE"], len(class_names), freeze=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["optimizer"]["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])

    model = train_model(dataloaders, model, criterion, optimizer, scheduler, config["DEVICE"], class_names, num_epochs=config["EPOCHS"])

torch.save(model.state_dict(), "data/best_model.pth")

test_image_pathes = np.array(glob(f"data/techosmotr/techosmotr/test/**/*.jpeg", recursive=True))
submission_df = pd.read_csv("data/sample_submission.csv")

for img_path in tqdm(test_image_pathes):
    id_ = int(img_path.split("/")[-1][:-5])
    pred = get_model_predictions(model, transforms["val"], config["DEVICE"], img_path)[0]
    submission_df.loc[submission_df["file_index"] == id_, "class"] = int(pred > 0)
submission_df.to_csv("output.csv", index=False)