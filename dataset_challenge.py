
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import config
from torchvision import transforms, utils
import requests

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_test_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_test_dataset(num_classes=5):
    tr = FoodDataset('train', num_classes)
    va = FoodDataset('val', num_classes)
    te = FoodDataset('test', num_classes)

    trans = transforms.Compose(
        [transforms.RandomRotation(90),
         transforms.RandomHorizontalFlip(),
         transforms.RandomPerspective(),
         transforms.ColorJitter()
         ]
    )

    convert = transforms.ToPILImage()
    count = 1

    for i, j in zip(tr.X,tr.y):
        dummy = convert(i)
        # dummy.show()
        #debug3 = trans(dummy)
        #debug3.show()
        #test = np.asarray(trans(dummy))
        #debug1 = i.shape
        #debug2 = test.shape
        tr.X = np.append(tr.X, [np.asarray(trans(dummy))], axis=0)
        tr.y = np.append(tr.y, [j], axis=0)
        print(count)
        if count > 999:
            break
        count = count + 1

    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)


    return tr, va, te, standardizer

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    # TODO: Complete this function
    count = 0
    image_dim = config('image_dim')
    resized = np.zeros((X.shape[0],image_dim,image_dim,3))
    for idx,i in enumerate(X):
        Im = Image.fromarray(i)
        resized[idx] = np.asarray(Im.resize((image_dim,image_dim), resample=Image.BICUBIC))
        count = count + 1
    print(count)
    #

    return resized

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = np.zeros(3)
        self.image_std = np.zeros(3)

    def fit(self, X):
        # TODO: Complete this function

        mean = np.zeros(3)
        std = np.zeros(3)

        mean[0] = np.mean(X[:,:,:,0])
        mean[1] = np.mean(X[:,:,:,1])
        mean[2] = np.mean(X[:,:,:,2])

        self.image_mean = mean

        std[0] = np.std(X[:,:,:,0])
        std[1] = np.std(X[:,:,:,1])
        std[2] = np.std(X[:,:,:,2])

        self.image_std = std



        #

    def transform(self, X):
        # TODO: Complete this function

        newX = X

        newX[:,:,:,0] = (X[:,:,:,0] - self.image_mean[0])/(self.image_std[0])
        newX[:, :, :, 1] = (X[:, :, :, 1] - self.image_mean[1]) / (self.image_std[1])
        newX[:, :, :, 2] = (X[:, :, :, 2] - self.image_mean[2]) / (self.image_std[2])

        return newX



        #

class FoodDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))

        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)

        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]

        X, y = [], []
        for i, row in df.iterrows():
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            y.append(row['numeric_label'])

        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'salad').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_test_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)

