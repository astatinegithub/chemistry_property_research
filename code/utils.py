import pandas as pd
from torch.utils.data import DataLoader


def create_dataloader(data_path, batch_size, IsShffle=True):
    data = pd.read_csv(data_path)
    data_loader = DataLoader(
        dataset=data["smiles"],
        batch_size=batch_size,
        shuffle=IsShffle
    )
    return data_loader


if __name__ == "__main__":
    # test
    path = "data/delaney-processed.csv"
    for i in create_dataloader(path, batch_size=32):
        print(i)