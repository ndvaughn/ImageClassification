import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder, AutoencoderClassifier
from train_common import *
from utils import config
import utils

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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

    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
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


def evaluate_autoencoder(dataset, model, criterion, get_semantic_label):
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
            y_pred.append(predicted)
            total = len(X)
            correct += (predicted == c).sum().item()
            val_acc = correct / total

        performance[c] = val_acc
        val_acc = 0.0

    for c, p in enumerate(performance):
        print('Class {}: {} accuracy'
            .format(get_semantic_label(c), p))


def main():
    # data loaders
    tr_loader, va_loader, te_loader, get_semantic_loader = get_train_val_test_loaders(
        num_classes=config('autoencoder.classifier.num_classes'))

    ae_classifier = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ae_classifier.parameters(),
        lr=config('autoencoder.classifier.learning_rate'))

    # freeze the weights of the encoder
    for name, param in ae_classifier.named_parameters():
        if 'fc1.' in name or 'fc2.' in name:
            param.requires_grad = False

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    ae_classifier, _, _ = restore_checkpoint(ae_classifier,
        config('autoencoder.checkpoint'), force=True, pretrain=True)
    print('Loading autoencoder classifier...')
    ae_classifier, start_epoch, stats = restore_checkpoint(ae_classifier,
        config('autoencoder.classifier.checkpoint'))

    fig, axes = utils.make_cnn_training_plot(name='Autoencoder Classifier')

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
        start_epoch, stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('autoencoder.classifier.num_epochs')):
        # Train model
        _train_epoch(tr_loader, ae_classifier, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
            epoch+1, stats)

        # Save model parameters
        save_checkpoint(ae_classifier, epoch+1,
            config('autoencoder.classifier.checkpoint'), stats)

    print('Finished Training')

    dataset = get_data_by_label(va_loader)

    ae_classifier, _, _ = restore_checkpoint(ae_classifier, config('autoencoder.classifier.checkpoint'))

    evaluate_autoencoder(dataset, ae_classifier, criterion, get_semantic_loader)

    # Keep plot open
    utils.save_cnn_training_plot(fig, name='ae_clf')
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
