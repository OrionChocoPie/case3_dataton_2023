import torch

config = dict(
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    EPOCHS = 30,
    RANDOM_STATE = 1465,
    N_SPLITS = 5,
    BATCH_SIZE = 32,
    N_WORKERS = 8,
    CLASS_WEIGHTS = [1, 3, 1, 2, 1.5],
    MODEL_NAME = "efficientnet_b1",
    optimizer = dict(
        lr = 1e-2,
    ),
    scheduler = dict(
        step_size = 7,
        gamma = 0.1,
    ),
)
