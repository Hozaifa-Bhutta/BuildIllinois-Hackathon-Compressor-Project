import os
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import numpy as np

image_size = 128
batch_size = 128
# number of channels in the iamge
nc = 3
stats = (0.5, ), (0.5, ) # normalizes values between -1-1, makes it more convenient for model training

device = torch.device("mps") 
dataset = datasets.ImageFolder("/Users/hozaifa/Documents/Coding projects/Sigaida/main_data/", transform = tt.Compose([
    tt.Resize(image_size), # resizes it to 'image_size'
    # tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats) # normalizes values between -1 to 1, makes it more convenient for model training
]))

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # batches training_dataset up and makes it iterable

# uncomment code below to display a plot of training images
example_batch = next(iter(train_dataloader))
plt.figure(figsize=(8,8)) # 8 by 8 grid
plt.title("Training Images")
plt.axis("off")
plt.imshow(np.transpose(make_grid(example_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

class Encoder(nn.Module):
    """This class is the template for our compressor module"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential (
            # input size: batch_size x 3 x 128 x 128
            nn.Conv2d(nc, 128, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(128), # normalizes data
            nn.ReLU(True),

            # input size: batch_size x 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(256), # normalizes data
            nn.ReLU(True),
            
            # input size: batch_size x 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True),

            # input size: batch_size x 512 x 16 x 16
            nn.Conv2d(512, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True),

            # input size: batch_size x 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True),

            # input size: batch_size x 512 x 4 x 4
            nn.Conv2d(512, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True),

            nn.Flatten()
            # output size: batch_size x 2048
        )
    def forward(self, input):
        return self.model(input) 
    

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.main = nn.Sequential(
            # input layer

            # input size: batch size x 2048
            nn.ConvTranspose2d(2048, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(65 * 8),
            nn.ReLU(True),

            # 1st hidden layer
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),

            # 2nd hidden layer
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),


            # 3rd hidden layer
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # output layer
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    

encoder = Encoder().to(device = torch.device("mps"))

# for batch_num, (images,_ ) in enumerate(train_dataloader):

#     images = images.to(torch.device("mps"))
#     print(f"image size {images.size()}")
#     encoded = encoder(images)
#     print(f"encoded size: {encoded.size()}")
#     decoded = decoder(encoded)
#     print(f"encoded size: {encoded.size()}")

#     break


# loading batch


# opt_
# encoded = encoder(batch)
# gen_image = decoder(encoded) # output the decoded value
# loss = nn.BCELoss()

# loss = (gen_image - batch)^2 * (0.5 + saliency_map )

# loss.backward()

# opt.step()