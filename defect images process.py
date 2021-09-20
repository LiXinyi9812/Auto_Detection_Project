import  torch
import  os, glob
import  random, csv

from  torch.utils.data import Dataset, DataLoader
from  torchvision import transforms
from PIL import Image

class Defectimg(Dataset):

    def __init__(self, root, resize, mode):
        super(Defectimg, self).__init__()
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