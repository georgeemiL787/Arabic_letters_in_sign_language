import os
from PIL import Image
from torch.utils.data import Dataset

class SingleFolderASLDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.samples = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Build class list from filenames
        self.class_names = sorted(list(set(self._extract_label(f) for f in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def _extract_label(self, filename):
        # Handles 'space_test.jpg', 'nothing_test.jpg', etc.
        if filename.lower().startswith('space'):
            return 'space'
        if filename.lower().startswith('nothing'):
            return 'nothing'
        return filename[0].upper()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self._extract_label(img_name)
        label_idx = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        return image, label_idx 