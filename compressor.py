import os
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
batch_size = 20
# number of channels in the iamge
nc = 3
stats = (0.5, ), (0.5, ) # normalizes values between -1-1, makes it more convenient for model training

device = torch.device("mps")
dataset = datasets.ImageFolder("../images", transform = tt.Compose([
    tt.Resize(image_size), # resizes it to 'image_size'
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats) # normalizes values between -1 to 1, makes it more convenient for model training
]))

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # batches training_dataset up and makes it iterable

# uncomment code below to display a plot of training images
# iterator = iter(train_dataloader)
# next(iterator)
# example_batch = next(iterator)
# # example_batch = train_dataloader.__iter__().__next__()
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
            nn.Conv2d(512, 512, kernel_size=4, stride =1, padding=1),
            nn.BatchNorm2d(512), # normalizes data
            nn.ReLU(True),
            nn.Flatten()
            # output size: batch_size x 4608
        )

        self.decoder = nn.Sequential(
            # input size: batch_size x 4608
            nn.ConvTranspose2d(4608, 512, kernel_size=4, stride=1, padding=0, bias=False), # upscales image  (can cause artifacts, look into this)
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size, 512, 4, 4;
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  

            # input size: batch_size, 256, 8, 8;
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size, 128, 16, 16;
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size, 64, 32, 32;
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size: batch_size, 32, 64, 64;
            nn.ConvTranspose2d(32, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
            # output size: batch_size x nc x 128 x 128
            

        )
    def forward(self, input):
        encoded = self.encoder(input)
        encoded_expanded = encoded[:, :, None, None]
        decoded = self.decoder(encoded_expanded)
        return decoded


compressor = Compressor().to(device = torch.device("mps"))

loss = nn.MSELoss(reduction='sum')

#write a cost function that compares the output image to the input image
def cost_function(output, input):
  saliency_map = make_saliency_maps(input).to(torch.device("mps"))
  diff = (output - input)
  diff *= saliency_map
  #convert diff to a 1d numpy array
  diff = torch.flatten(diff)
  return torch.dot(diff, diff)

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

def make_grayscale(array):
  #array is a 3 by 128 by 128 tensor
  return 0.299 * array[0] + 0.587 *array[1] + 0.114 * array[2]

def make_saliency_maps(images: torch.Tensor):
  #image is a 3 by 128 by 128 tensor
  allSaliencyMaps = np.ndarray((images.shape[0], 3, 128, 128), dtype=np.float32)
  numImages = images.shape[0]
  for idx in range(numImages):
    image = images[idx]
    array = image.cpu().numpy()
    (success, saliencyMap) = saliency.computeSaliency(make_grayscale(array))
    allSaliencyMaps[idx][0] = saliencyMap
    allSaliencyMaps[idx][1] = saliencyMap
    allSaliencyMaps[idx][2] = saliencyMap
  return torch.tensor(allSaliencyMaps)


num_epochs = 10

optimizer = torch.optim.Adam(compressor.parameters(), lr = 10e-3, betas=(0.5, 0.999)) # defines optimizer
loss_function = nn.MSELoss(reduction="sum") # mean squared error loss
size = len(train_dataloader) # number of batches
image = train_dataloader.__iter__().__next__()[0]
print(image.size())
# exit(0)
# for epoch in range(num_epochs*1000):
#     print(f"Epoch: {epoch}")
#     print("######################################")
#     # for batch_num, (images,_ ) in enumerate(train_dataloader):
#       # print(f"batch number: {batch_num}/{size}")
#     image = image.to(torch.device("mps"))
#     encoded_image = compressor(image) # generates images

#     #  compute mean squared error
#     loss = loss_function(image, encoded_image)
#     loss.backward()
#     optimizer.step()
#     print(loss)

#     print(f"loss: {loss.item()}")

# torch.save(compressor.state_dict(), "compressor.pth")
for images in train_dataloader:
  images = images[0].to(torch.device("mps"))
  make_saliency_maps(images)
  # break
  print(f"image size {images.size()}")
  # print(images)
  encoded = compressor(images)
  print(f"encoded size: {encoded.size()}")
  #make a saliency map of all ones
  saliency_map = np.ones(128 * 3 * 128 * 128)
  print(cost_function(encoded, images))
  plt.figure(figsize=(8,8)) # 8 by 8 grid
  plt.title("Output Images")
  plt.axis("off")
  plt.imshow(np.transpose(make_grid(encoded.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
  plt.show()

  break

# loading batch


# opt_
# encoded = encoder(batch)
# gen_image = decoder(encoded) # output the decoded value
# loss = nn.BCELoss()

# loss = (gen_image - batch)^2 * (0.5 + saliency_map )

# loss.backward()

# opt.step()
