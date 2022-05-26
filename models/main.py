from data_loader import get_data_loaders
from model import ConvolutionalNetwork
from flow import test, train
import torch.optim as optim
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
train_data, test_data = get_data_loaders()
model = ConvolutionalNetwork().to(device)
lr = 1e-1
epochs = 3
log_intervals = 10
optimizer = optim.SGD(model.parameters(), lr=lr)

train_losses, train_counter = train(model, optimizer, epochs, train_data, device, log_intervals)
test(model, test_data, device)
