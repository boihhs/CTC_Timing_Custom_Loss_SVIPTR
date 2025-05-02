import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image\

"""
Written by Leo Benaharon on 4/16/2025
Read datasets with a word file showing tuples of image location and ground truth text respecfully
"""

vocab = {
    ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
    'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '1': 27, '2': 28, '3': 29, '4': 30,
    '5': 31, '6': 32, '7': 33, '8': 34, '9': 35, '0': 36, '?': 37, "'": 38, '!': 39, '>': 40,
    ':': 41, '.': 42, ')': 43, '$': 44, '-': 45, ',': 46, '(': 47, '"': 48, '#': 49, '%': 49,
    '&': 49, '*': 49, '+': 49, '/': 49, ';': 49, '<': 49, '=': 49, '@': 49, '[': 49, '\\': 49,
    ']': 49, '^': 49, '_': 49, '`': 49, '{': 49, '|': 49, '}': 49, '~': 49
}

vocab_reverse = {
  0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
  11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
  21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '1', 28: '2', 29: '3', 30: '4',
  31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: '0', 37: '?', 38: "'", 39: '!', 40: '>',
  41: ':', 42: '.', 43: ')', 44: '$', 45: '-', 46: ',', 47: '(', 48: '"', 49: '@'
}

class ImageTextDataset2(Dataset):
    def __init__(self, annotations_file, img_dir, vocab):

        self.img_dir = img_dir
        self.vocab = vocab
        # read annotations file
        self.samples = []  # list of (img_path, text)
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, tag = line.split(',', 1)
                path = path.strip()
                tag = tag.strip().strip('"').lower()
                if tag == '':
                    tag = ' '
                
                # filter out bad ones
                if len(tag) > 15:
                    continue
                if any(c not in vocab for c in tag):
                    continue
                self.samples.append((path, tag))
        print(len(self.samples))

        # For training
        # self.transform = transforms.Compose([
        #     transforms.ColorJitter(
        #         brightness=0.1,
        #         contrast=0.1,
        #         saturation=0.1,
        #         hue=0.05
        #     ),
        #     transforms.RandomPerspective(
        #         distortion_scale=0.1,
        #         p=0.2
        #     ),
        #     transforms.RandomAffine(
        #         degrees=5,
        #         translate=(0.05, 0.05),
        #         scale=(0.95, 1.05),
        #         shear=5
        #     ),
        #     transforms.Resize((32, 96)),
        #     transforms.ToTensor()
        # ])

        # For Testing
        self.transform = transforms.Compose([
            transforms.Resize((32, 96)),
            transforms.ToTensor()
        ])
        self.target_transform = self.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, label_str = self.samples[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(img_rel))
        image = Image.open(img_path).convert('RGB')
        # Now do the same as before
        if len(label_str) > 15:
            label_str = label_str[:15]

        label = [self.vocab[str(char)] for char in label_str]

        new_label = [0]  # Start with a 0
        for value in label:
            new_label.append(value)
            new_label.append(0)  # Add a 0 after each element
        
        size = len(new_label) # Get the size

        # Convert everything to the same length
        desired_length = 49
        if len(new_label) < desired_length:
            padded_label = new_label + [0] * (desired_length - len(new_label))

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(padded_label), torch.tensor(size, dtype=torch.long)
    

# Same as before
def collate_fn(batch):
    # Get lists of the images and labels
    images, labels, sizes = zip(*batch)
    #images, labels = zip(*batch)

    # Stack the images to get (B x H x W x C)
    images_out = torch.stack(images)

    B, _, _, _ = images_out.shape

    # Get the length of each vector in the list
    #lengths_out = torch.tensor([i.size(0) for i in labels], dtype=torch.long)
    lengths_out = torch.stack(sizes)

    #labels_out = torch.cat(labels, dim=0)
    labels_out = torch.stack(labels)

    #return images_out.permute(0, 2, 3, 1), lengths_out, labels_out, torch.ones(B, dtype=torch.long) * 24
    return images_out.permute(0, 2, 3, 1), lengths_out, labels_out


