import torch.nn.functional as F
import torch


def train(model, optimizer, epochs, data_loader, device, log_interval):
    train_losses = []
    train_counter = []
    model.train()
    for epoch in range(epochs):
        for batch_idx, (train_value, train_label) in enumerate(data_loader):
            train_value = train_value.to(device)
            test_label = train_label.to(device)
            optimizer.zero_grad()
            yhat = model(train_value)
            loss = F.nll_loss(yhat, test_label)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(train_value), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss))
                train_losses.append(loss)
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(data_loader.dataset)))

    return train_losses, train_counter


def test(model, data_loader, device):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
