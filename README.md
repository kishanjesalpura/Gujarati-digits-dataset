# Gujarati Handwritten Digits Dataset

![Gujarati Handwritten Digits](https://github.com/kishanjesalpura/Gujarati-digits-dataset/)

## Overview
The Gujarati Handwritten Digits Dataset is an MNIST-like dataset specifically curated for handwritten digits in Gujarati. It contains 1000 images, with 100 images for each digit from 0 to 9.

## Description
- **Dataset Size**: This dataset comprises 1000 handwritten digit images.
- **Format**: Images are stored in PNG format.
- **Resolution**: All images are 28x28 pixels and are black and white.
- **File Structure**: The images for each digit are organized into corresponding folders, and each image is named in the format "{digit}_{index}.png", where "digit" represents the digit and "index" represents the index of the image (e.g., 0_98.png is the 98th image of digit 0).
- **Label Mapping**: A CSV file with mappings of image names to their corresponding labels is provided. The image names are relative to the parent directory "Dataset".

## Purpose
The primary objective of this dataset is to serve as a valuable resource for enthusiasts interested in machine learning, computer vision, and pattern recognition tasks specific to handwritten Gujarati digits.

## Contribution Guidelines
Contributions to this dataset are welcome! Feel free to submit a pull request if you have additional data, corrections, or improvements.

## Demo Python script to use this dataset in pytorch

### Class definition for dataset

```python
from torch.utils.data import Dataset, DataLoader
from skimage import io
import pandas as pd

## Class definition

class GMNIST(Dataset):
  """ MNIST like database for Hindi Characters"""

  def __init__(self, csv_file, root_dir, transform = None):
    """
    Arguments:
      csv_file (string): Path to csv file with labels.
      root_dir (string): Directory with all the images.
      transform (callable, optional): Optional transform to be applied on a sample.
    """

    self.labels = pd.read_csv(csv_file, header=None)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
    image = io.imread(img_name)
    image = image.astype(np.float32)

    label = self.labels.iloc[idx, 1]

    if self.transform:
      image = self.transform(image)

    return image, label
```
