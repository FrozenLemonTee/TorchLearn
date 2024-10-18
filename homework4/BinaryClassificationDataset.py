import os.path

from PIL import Image
from torch.utils.data import Dataset

from homework4.dataSplit import get_img_names


class BinaryClassificationDataset(Dataset):
    def __init__(self, data_dir, labels, transform):
        self.data_dir = data_dir
        self.transform = transform
        self._labels = {labels[0]: 0, labels[1]: 1}
        self.data_info = self._get_img_info(data_dir)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_name, label = self.data_info[index]
        img = Image.open(os.path.join(self.data_dir, f"{img_name}")).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def _get_img_info(self, data_dir):
        data_info = []
        img_names = get_img_names(data_dir)
        for img_name in img_names:
            data_info.append([img_name, self._labels[str(img_name).split('.')[0]]])

        return data_info
