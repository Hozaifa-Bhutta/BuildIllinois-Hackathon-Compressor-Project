import argparse, sys
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
from torch.quantization import DeQuantStub, QuantStub
if len(os.sys.argv) < 2:
    print("Please type in your path and whether you wish to decode or encode")
    exit()

parser=argparse.ArgumentParser()
parser.add_argument("--Decode", help="sets decode to true or false")
parser.add_argument("--Encode", help="Sets encode to true or false")
parser.add_argument("--EncodePath", help="Sets encode path")
parser.add_argument("--DecodePath", help="Sets decode path")

args=parser.parse_args()

DECODE = args.Decode
DECODE_PATH = args.DecodePath

ENCODE = args.Encode
ENCODE_PATH = args.EncodePath

if (DECODE and ENCODE):
    print("Error, only set encode or decode to true")
    exit()


image_size = 128
batch_size = 1
# number of channels in the iamge
nc = 3
stats = (0.5, ), (0.5, ) # normalizes values between -1-1, makes it more convenient for model training
device = "cpu"



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
    
    def encode(self, input):
        return self.encoder(input) # just returns the encoded version
    
    def decode(self, input):
        encoded_expanded = input[:, :, None, None] # treats the vector as an image
        return self.decoder(encoded_expanded) # just returns the decoder version
    

model = Compressor()
model.load_state_dict(torch.load("compressor.pth"))
model.to(device = device)

if ENCODE:
    dataset = datasets.ImageFolder(ENCODE_PATH, transform = tt.Compose([
        tt.Resize(image_size), # resizes it to 'image_size'
        tt.CenterCrop(image_size),
        tt.ToTensor(),
        tt.Normalize(*stats) # normalizes values between -1 to 1, makes it more convenient for model training
    ]))
    # this encodes all the images
    for batch_num, (image, _) in enumerate(dataset):
        image = image.to(device)
        image = image[None, :, :, :] # adds dimension for model
        encoded_value = model.encode(image)

        # converts to numpy array
        encoded_array = encoded_value.detach().cpu().numpy()

        # save array locally
        np.save(f"encodedImage", encoded_array)
        print(f"Successfully enoded image as a numpy array of size {encoded_array.size}")

if DECODE:
    # decodes image and saves it
    # loads numpy array
    loaded_numpy_array = np.load(DECODE_PATH)
    encoded_val = torch.from_numpy(loaded_numpy_array)
    generated_image = model.decode(encoded_val).to(device = device)
    plt.figure(figsize=(1,1)) # 1 by 1 grid
    plt.imshow(np.transpose(make_grid(generated_image.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.box(False)
    plt.savefig("Decompressed.png", bbox_inches="tight", pad_inches = 0)
    print("Successfully loaded image into hardrive")
