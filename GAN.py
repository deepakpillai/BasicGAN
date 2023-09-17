import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def show(tensor: torch.Tensor, ch=1, size=(28,28), num=16):
  # data = tensor.detach().cpu().view(-1, 1, 28, 28)
  data = tensor.detach().cpu().view(-1, ch, *size)
  grid = make_grid(data[:num], nrow=4).permute(1,2,0)
  plt.figure(figsize=(10,5))
  plt.imshow(grid)
  plt.show()

def show_side_by_side(fake_tensor: torch.Tensor, real_tensor: torch.Tensor, ch=1, size=(28,28), num=16):
  fake_data = fake_tensor.detach().cpu().view(-1, ch, *size)
  fake_grid = make_grid(fake_data[:num], nrow=4).permute(1,2,0).cpu()

  real_data = real_tensor.detach().cpu().view(-1, ch, *size)
  real_grid = make_grid(real_data[:num], nrow=4).permute(1,2,0).cpu()

  fig = plt.figure(figsize=(8,4))
  plt.subplot(1,2,1)
  plt.imshow(fake_grid)
  plt.subplot(1,2,2)
  plt.imshow(real_grid)
  plt.show()

noice = torch.randn(128, 784).to(device)
show_side_by_side(noice, noice)

noice = torch.randn(128, 784)
show(noice)

def create_noice(batch_size, out_features):
  # return torch.randn(128, 784).to(device)
  return torch.randn(batch_size, out_features).to(device)

def gen_block(in_feauures, out_features):
  return nn.Sequential(
      nn.Linear(in_features=in_feauures, out_features=out_features),
      nn.BatchNorm1d(out_features),
      nn.ReLU(inplace=True)
  )

class Generator(nn.Module):
  def __init__(self, in_features = 64, out_features = 784, hidden_layers_dim = 128):
    super().__init__()
    self.gen = nn.Sequential(
        gen_block(in_feauures=in_features, out_features=hidden_layers_dim), #64 -> 128
        gen_block(in_feauures=hidden_layers_dim, out_features=hidden_layers_dim * 2), #128 -> 256
        gen_block(in_feauures=hidden_layers_dim * 2, out_features=hidden_layers_dim * 4), #256 -> 512
        gen_block(in_feauures=hidden_layers_dim * 4, out_features=hidden_layers_dim * 8), #512 -> 1024
        nn.Linear(in_features = hidden_layers_dim * 8, out_features=out_features), #1024 -> 784 (784 is 28*28. 28*28 is the size of MINST dataset image width and height)
        nn.Sigmoid()
    )

  def forward(self, noice):
    return self.gen(noice)

def disc_block(in_features, out_features):
  return nn.Sequential(
      nn.Linear(in_features=in_features, out_features=out_features),
      nn.LeakyReLU(0.2)
  )

class Discriminator(nn.Module):
  def __init__(self, in_features = 784, out_features = 1, hidden_layer_dim = 256):
    super().__init__()
    self.disc = nn.Sequential(
        disc_block(in_features, hidden_layer_dim * 4), #784 -> 1024
        disc_block(hidden_layer_dim * 4, hidden_layer_dim * 2), #1024 -> 512
        disc_block(hidden_layer_dim * 2, hidden_layer_dim), #512 -> 256
        nn.Linear(in_features=hidden_layer_dim, out_features=1) #256 -> 1
    )

  def forward(self, image):
    return self.disc(image)

epochs = 600
curent_step = 0
info_required_batch_step = 300
mean_gen_loss = 0
mean_disc_loss = 0

gen_in_features = 64
lr = 0.0001
batch_size = 128

dataset = MNIST(".", download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

x, y = next(iter(dataloader))
print(x.shape, y.shape)

gen = Generator(in_features=gen_in_features).to(device)
disc = Discriminator().to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
gen_optim = torch.optim.Adam(gen.parameters(), lr = lr)
disc_optim = torch.optim.Adam(disc.parameters(), lr = lr)

data = create_noice(batch_size, gen_in_features) #batch_size = 128, gen_in_features = 64
print(data.shape) #output: torch.Size([128, 64])

fake_image = gen(data) #this create an potput of shape torch.Size([128, 784])
print(fake_image.shape)

show(fake_image)

#understanding ones_like and ones method
test = torch.randn(3)
print(test)
test_once = torch.ones_like(test) #ones_like create a tensor similer to the input tensor shape
print(test_once)

print(torch.ones(1, 3)) #ones create a tensor based on the input shape

def calc_generator_loss(loss_fn,  generator, discriminator, batch_size, in_features):
  noice = create_noice(batch_size, in_features)
  fake_image = generator(noice)
  disc_pred = discriminator(fake_image)
  targets = torch.ones_like(disc_pred) #targets == labels
  generator_loss = loss_fn(disc_pred, targets)
  return generator_loss

def calc_discriminator_loss(loss_fn, generator, discriminator, batch_size, in_features, real_image):
  noice = create_noice(batch_size, in_features)
  fake_image = generator(noice)
  discriminaor_fake = discriminator(fake_image.detach())

  #detach() is a PyTorch function that is used to detach a tensor from the computation graph. When we perform operations on a tensor,
  #PyTorch builds a computation graph that tracks the operations performed on the tensor. The detach() function is used to remove a tensor from the computation graph,
  #making it a standalone tensor that is no longer linked to the graph.

  discriminator_fake_targets = torch.zeros_like(discriminaor_fake)
  discriminator_fake_loss = loss_fn(discriminaor_fake, discriminator_fake_targets)

  discriminator_real = discriminator(real_image)
  discriminator_real_targets = torch.ones_like(discriminator_real)
  discriminator_real_loss = loss_fn(discriminator_real, discriminator_real_targets)

  discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

  return discriminator_loss

for epoch in tqdm(range(epochs)):
  for real_image, _ in dataloader:

    #Discriminator
    disc_optim.zero_grad()
    batch_size = len(real_image)
    real = real_image.view(batch_size, -1).to(device)

    disc_loss = calc_discriminator_loss(loss_fn, gen, disc, batch_size, gen_in_features, real)
    disc_loss.backward(retain_graph = True)
    disc_optim.step()

    #Generator
    gen_optim.zero_grad()
    gen_loss = calc_generator_loss(loss_fn, gen, disc, batch_size, gen_in_features)
    gen_loss.backward(retain_graph = True)
    gen_optim.step()

    #Visualise Visualise Visualise
    mean_gen_loss += gen_loss.item() / info_required_batch_step
    mean_disc_loss += disc_loss.item() / info_required_batch_step

    if curent_step % info_required_batch_step == 0 and curent_step > 0:
      fake_noice = create_noice(batch_size, gen_in_features)
      fake_image = gen(fake_noice)
      # show_side_by_side(fake_image, real)
      print(f"\nEpoch: {epoch}, Current step: {curent_step}, mean_gen_loss: {mean_gen_loss}, mean_disc_loss: {mean_disc_loss}")
      mean_gen_loss = 0
      mean_disc_loss = 0

    curent_step+=1

torch.save(gen, "./gen.pth")
torch.save(disc, "./disc.pth")

torch.save(gen.state_dict(), "./gen_s.pth")
torch.save(disc.state_dict(), "./disc_s.pth")

os.system("shutdown /s /t 1")