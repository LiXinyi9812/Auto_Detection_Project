import  torch
import  os, glob
import  random, csv

from  torch.utils.data import Dataset, DataLoader
from  torchvision import transforms
from PIL import Image

#optimization1：imblearn smote
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE







class RLBD(Dataset):

    def __init__(self, root, resize, mode):
        super(RLBD, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())



        self.images, self.labels = self.load_csv('images.csv')


        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.bmp'))
            print(len(images), images)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode = 'w', newline = '') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels


    def __len__(self):
        return len(self.images)

    def denormalize(self,x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat*std + mean

        return x

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label


def main():
    import visdom
    import time
    import torchvision

    viz = visdom.Visdom()

    db = RLBD('RLBD', 64, 'train')

    x,y = next(iter(db))


    # smo = SMOTE(random_state=42)
    # X_smo, y_smo = smo.fit_sample(x, y)
    # from collections import Counter
    # print(Counter(y_smo))

    viz.image(db.denormalize(x),win='sample_x',opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    for x,y in loader:
        viz.images(db.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
        time.sleep(10)






if __name__ == '__main__':
    main()