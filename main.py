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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
RANDOM_STATE = 1465
N_SPLITS = 5

image_pathes = np.array(glob(f"data/techosmotr/techosmotr/train/**/*.jpeg", recursive=True))
image_classes = np.array([image_path.split("/")[-2] for image_path in image_pathes])
class_names = sorted(set(image_classes))

index2class = {i: x for i, x in enumerate(class_names)}
class2index = {x: i for i, x in index2class.items()}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = {}
for i, (train_index, val_index) in enumerate(skf.split(image_pathes, image_classes)):
    folds[i] = (train_index, val_index)

for index_fold, fold in folds.items():
    print(f"Run on {index_fold} fold...")
    indexes = {"train": fold[0], "val": fold[1]}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            CarsDataset(image_pathes, image_classes, indexes[x], class2index, transforms[x]),
            batch_size=32, 
            shuffle=True, 
            num_workers=8,
        )
        for x in ['train', 'val']
    }

    class_weights = torch.tensor([1, 3, 1, 2, 1.5], dtype=torch.float).to(DEVICE)

    model = get_model("efficientnet_b1", DEVICE, len(class_names), freeze=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(dataloaders, model, criterion, optimizer, scheduler, DEVICE, class_names, num_epochs=EPOCHS)

torch.save(model.state_dict(), "data/best_model.pth")

test_image_pathes = np.array(glob(f"data/techosmotr/techosmotr/test/**/*.jpeg", recursive=True))
submission_df = pd.read_csv("data/sample_submission.csv")

for img_path in tqdm(test_image_pathes):
    id_ = int(img_path.split("/")[-1][:-5])
    pred = get_model_predictions(model, transforms["val"], DEVICE, img_path)[0]
    submission_df.loc[submission_df["file_index"] == id_, "class"] = int(pred > 0)
submission_df.to_csv("output.csv", index=False)