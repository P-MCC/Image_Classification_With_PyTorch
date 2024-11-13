from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class PlayingCardDataset(Dataset):
    """
    A custom dataset class for loading playing card images from a directory with optional transformations.
    """
    def __init__(self, data_dir, transform=False):
        """
        Initializes the dataset by loading images from the given directory using ImageFolder.

        Args:
            data_dir (str): Path to the root directory containing the dataset.
            use_transform (bool): If True, applies default transformations to the images.
        """
        # Default transformations if use_transform is True
        if transform:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
        else:
            self.transform = None

        # Initialize ImageFolder with the data directory
        self.data = ImageFolder(data_dir)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the image and its label by index and applies the transform if provided.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        image, label = self.data[idx]

        # Apply the transformation if it's set
        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        """Returns the list of class labels."""
        return self.data.classes

    @property
    def class_to_idx(self):
        """Returns the class-to-index mapping."""
        return self.data.class_to_idx

    @property
    def img_size(self):
        """Returns the size of the images in the dataset."""
        try:
            image_path, _ = self.data.imgs[0]
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print("Could not retrieve image size:", e)
            return None
        
dataset = PlayingCardDataset(data_dir="M:/Datasets/Playing_Card", transform=True)
image, label = dataset[100]

# Check the shape of the retrieved image tensor
image_shape = image.shape
print(image_shape)