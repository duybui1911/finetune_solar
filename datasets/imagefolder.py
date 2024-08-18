import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageOps

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        padding = (pad_left, pad_top, max_wh - w - pad_left, max_wh - h - pad_top)
        return ImageOps.expand(image, padding, fill=255)  # Use white padding


    
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, train_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, where subfolders are class labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        print(self.class_to_idx)
        self.samples = self._make_dataset(self.root_dir, self.class_to_idx)
        self.train_transform = train_transform
        self.square = transform = transforms.Compose([SquarePad(), transforms.Resize((500, 500))])
    def _find_classes(self, dir):
        """Find the class folders in a dataset."""
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        """Make the dataset by listing all image file paths and their corresponding labels."""
        images = []
        for target_class in sorted(class_to_idx.keys()):
            class_idx = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append((path, class_idx))
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image  =self.square(image)
        if self.train_transform:
            image = self.train_transform(image)
        if self.transform:
            image = self.transform(image)

        return image, label
