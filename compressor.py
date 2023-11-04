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
num_epochs = 100
lr = 0.0002

device = torch.device("mps") 
dataset = datasets.ImageFolder("/Users/hozaifa/Documents/Coding projects/Sigaida/main_data/", transform = tt.Compose([
    tt.Resize(image_size), # resizes it to 'image_size'
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats) # normalizes values between -1 to 1, makes it more convenient for model training
]))

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # batches training_dataset up and makes it iterable

# uncomment code below to display a plot of training images
# example_batch = next(iter(train_dataloader))
# plt.figure(figsize=(8,8)) # 8 by 8 grid
# plt.title("Training Images")
# plt.axis("off")
# plt.imshow(np.transpose(make_grid(example_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

class Compressor(nn.Module):
    """This class is the template for our compressor module"""
    def __init__(self):
        super(Compressor, self).__init__()
        self.encoder = nn.Sequential (
            # input size: batch_size x 3 x 128 x 128
            nn.Conv2d(nc, 128, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(128), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size x 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(256), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),
            
            # input size: batch_size x 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size x 512 x 16 x 16
            nn.Conv2d(512, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size x 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=4, stride =2, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size x 512 x 4 x 4
            nn.Conv2d(512, 512, kernel_size=4, stride =1, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.LeakyReLU(0.2, inplace=True),


            nn.Flatten()
            # output size: batch_size x 4608
        )

        self.decoder = nn.Sequential(
            # input size: batch_size x 4608
            nn.ConvTranspose2d(4608, 512, kernel_size=4, stride=1, padding=0, bias=False), # upscales image  (can cause artifacts, look into this)
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s

            # input size: batch_size, 512, 4, 4;
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s

            # input size: batch_size, 256, 8, 8;
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s


            # input size: batch_size, 128, 16, 16;
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s


            # input size: batch_size, 64, 32, 32;
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s


            # input size: batch_size, 32, 64, 64;
            nn.ConvTranspose2d(32, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # output size: batch_size x nc x 128 x 128
            

        )
    def forward(self, input):
        encoded = self.encoder(input)
        encoded_expanded = encoded[:, :, None, None] # treats the vector as an image
        decoded = self.decoder(encoded_expanded)
        return decoded
    


compressor = Compressor().to(device = device)

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def cost_function(output, input):

  diff = torch.flatten(output - input) # subtract them from each other
  return torch.dot(diff, diff) # dot product with itself gives the square of the sum of the components

optimizer = torch.optim.Adam(compressor.parameters(), lr = lr, betas=(0.5, 0.999)) # defines optimizer
loss_function = nn.MSELoss(reduction="sum") # mean squared error loss
size = len(train_dataloader) # number of batches
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    print("######################################")
    for batch_num, (images,_ ) in enumerate(train_dataloader):
        print(f"batch number: {batch_num}/{size}")
        images = images.to(torch.device("mps"))
        encoded_images = compressor(images) # generates images

        #  compute mean squared error
        loss = loss_function(images, encoded_images)
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")




# loading batch


# opt_
# encoded = encoder(batch)
# gen_image = decoder(encoded) # output the decoded value

# loss = (gen_image - batch)^2 * (0.5 + saliency_map )

# loss.backward()

# opt.step()
