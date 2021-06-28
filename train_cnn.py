import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.cnn import CNN
from train_common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_data_by_label(dataset):
    data = []
    for i, (X, y) in enumerate(dataset):
        for c in range(config('autoencoder.num_classes')):
            batch = X[(y == c).nonzero().squeeze(1)]
            if len(data) <= c:
                data.append(batch)
            else:
                data[c] = torch.cat((data[c], batch))
    return data


def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch,
    stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in tr_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    train_loss = np.mean(running_loss)
    train_acc = correct / total
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in val_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)


def evaluate_cnn(dataset, model, criterion, get_semantic_label):
    """
    Runs inference on an autoencoder model to evaluate the mse loss on the
    validation sets. Reports per-class performance to terminal.
    """

    num_classes = config('autoencoder.num_classes')
    batch_size = config('autoencoder.batch_size')
    performance = np.zeros(num_classes)
    for c in range(num_classes):
        with torch.no_grad():
            y_true, y_pred = [], []
            correct, total = 0, 0
            X = dataset[c]
            output = model(X)
            predicted = predictions(output.data)
            total = len(X)
            correct += (predicted == c).sum().item()
            val_acc = correct / total

        performance[c] = val_acc
        val_acc = 0.0

    for c, p in enumerate(performance):
        print('Class {}: {} accuracy'
            .format(get_semantic_label(c), p))


def main():
    # Data loaders
    tr_loader, va_loader, te_loader, get_semantic_labels = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))

    # Model
    model = CNN()

    # TODO: define loss function, and optimizer
    params = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters())
    params = params + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=0.0001)
    #

    print('Number of float-valued parameters:', count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print('Loading cnn...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('cnn.checkpoint'))

    fig, axes = utils.make_cnn_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('cnn.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('cnn.checkpoint'), stats)

    print('Finished Training')

    model,_ ,_ = restore_checkpoint(model, config('cnn.checkpoint'))

    dataset = get_data_by_label(va_loader)
    evaluate_cnn(dataset, model, criterion, get_semantic_labels)

    # Save figure and keep plot open
    utils.save_cnn_training_plot(fig)
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
