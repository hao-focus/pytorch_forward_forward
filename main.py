import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=False,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=False,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        # self.layers = []

        self.layer1 = Layer(dims[0], dims[1]).cuda()
        self.layer2 = Blackbox_layer(dims[1], dims[2], device=device).cuda()

        # for d in range(len(dims) - 1):
        #     self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            print('predicting label', label, '...')
            h = overlay_y_on_x(x, label)
            goodness = []
            # for layer in self.layers:
            #     h = layer(h) # [50000, 500]
            #     goodness += [h.pow(2).mean(1)] # list 2: [50000]
            h = self.layer1(h)
            goodness += [h.pow(2).mean(1)]
            h = self.layer2(h)
            goodness += [h.pow(2).mean(1)]

            goodness_per_label += [sum(goodness).unsqueeze(1)] # list 10: [50000, 1]
        goodness_per_label = torch.cat(goodness_per_label, 1) # [50000, 10]
        return goodness_per_label.argmax(1) # [50000]

    def train(self, x_pos, x_neg):
        # h_pos, h_neg = x_pos, x_neg
        # for i, layer in enumerate(self.layers):
        #     print('training layer', i, '...')
        #     h_pos, h_neg = layer.train(h_pos, h_neg)

        print('training layer 1...')
        h_pos, h_neg = self.layer1.train(x_pos, x_neg)
        # print('forwarding blackbox layer...')
        # h_pos, h_neg = self.layer2.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.sigmoid(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        # return self.relu(
        #     torch.mm(x_direction, self.weight.T) +
        #     self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1) # [50000]
            g_neg = self.forward(x_neg).pow(2).mean(1) # [50000]
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

class Blackbox_layer(nn.Module):
    """
    This class is a blackbox layer that are not optimized once defined in the middle of the network.
    """
    def __init__(self, in_features, out_features, device=None):
        super(Blackbox_layer, self).__init__()
        self.device = device
        torch.manual_seed(42)
        tm_real = torch.normal(mean=0, std=1.0/torch.sqrt(torch.tensor(in_features)), size=(in_features, out_features)).to(self.device)
        tm_imag = torch.normal(mean=0, std=1.0/torch.sqrt(torch.tensor(in_features)), size=(in_features, out_features)).to(self.device)
        tm = tm_real + 1j*tm_imag
        self.tm = torch.nn.Parameter(tm, requires_grad=False)

    def forward(self, x):
        # x = , since the output from the previous layer is in the range of [0, 1], I don't need to normalize it right now.
        x = torch.exp(1.0j * math.pi * x) # [0, pi]
        x = torch.mm(x, self.tm)
        x = torch.square(torch.abs(x))
        x = x / torch.max(x, dim=1, keepdim=True)[0]
        return x

    def train(self, x_pos, x_neg):
        """
        This function is for matching the interface of the Layer class.
        """
        x_pos = self.forward(x_pos)
        x_neg = self.forward(x_neg)
        return x_pos.detach(), x_neg.detach() # detach() is used to prevent the gradient from flowing back to the blackbox layer.
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader)) # x: [50000, 784], y: [50000]
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
