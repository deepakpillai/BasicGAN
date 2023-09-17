import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show(tensor: torch.Tensor, ch=1, size=(28,28), num=16):
  # data = tensor.detach().cpu().view(-1, 1, 28, 28)
  data = tensor.detach().cpu().view(-1, ch, *size)
  grid = make_grid(data[:num], nrow=4).permute(1,2,0)
#   plt.figure(figsize=(10,5))
  plt.imshow(grid)
  plt.show()


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
  
model = Generator().eval()
model.load_state_dict(torch.load("./Model/state_dict/gen_s.pth"))
# print(model)

noice = torch.randn(128, 64)
pred = model(noice)
show(pred)