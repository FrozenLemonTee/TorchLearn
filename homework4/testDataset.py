import os

from torch.utils.data import DataLoader
from torchvision import transforms

from RMBdataset import RMBDataset

if __name__ == '__main__':
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, 'RMB_data')
    RMBDataset = RMBDataset(dataset_dir, transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]))
    RMBDataLoader = DataLoader(RMBDataset, batch_size=1, shuffle=True)
    for i, data in enumerate(RMBDataLoader):
        print(i, data)