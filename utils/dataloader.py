from torch.utils.data import Dataset
from PIL import Image
import os

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # 第二個值是 dummy label
