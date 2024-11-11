from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class PlayingCardDataset(Dataset):
    """
    A custom dataset class for loading playing card images from a directory.

    Args:
        data_dir (str): The path to the dataset directory containing images organized in subfolders.
        transform (callable, optional): A function/transform to apply to the images (e.g., data augmentation, normalization).

    Attributes:
        data (ImageFolder): A torchvision ImageFolder dataset object.
        transform (callable): A transform to be applied to the images during data loading.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset by loading images from the given directory using ImageFolder.

        Args:
            data_dir (str): Path to the root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Initialize the dataset using ImageFolder from torchvision
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Fetches the image and its label by index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        return self.data[idx]

    @property
    def classes(self):
        """
        Returns the list of class labels.

        Returns:
            list: List of class names.
        """
        return self.data.classes

    @property
    def class_to_idx(self):
        """
        Returns the class-to-index mapping.

        Returns:
            dict: A dictionary mapping class names to indices.
        """
        return self.data.class_to_idx

    @property
    def img_size(self):
        """
        Returns the size of the images in the dataset.

        Returns:
            tuple: (width, height) of the images.
        """
        # Try getting image size from the first image in the dataset
        try:
            image_path, _ = self.data.imgs[0]
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print("Could not retrieve image size:", e)
            return None