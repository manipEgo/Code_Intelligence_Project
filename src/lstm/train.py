import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import CodeDataset
from logger import get_logger
from model import LSTMModel
from torch.utils.data import DataLoader


def load_data(train_filepath, test_filepath, token2idx):
    train_dataset = CodeDataset(train_filepath, token2idx, 8)
    test_dataset = CodeDataset(test_filepath, token2idx, 2)
    train_loader = DataLoader(train_dataset, batch_size=OPT.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=OPT.batch_size, shuffle=True)
    return train_loader, test_loader


def train(model: LSTMModel, train_loader: DataLoader, test_loader: DataLoader):
    if OPT.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = OPT.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=OPT.learning_rate)

    for epoch in range(OPT.max_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            h0, c0 = model.init_hidden(OPT.batch_size)
            optimizer.zero_grad()

            outputs, hidden = model(inputs, (h0, c0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


def tokenize(filepath):
    tokens = []
    with open(filepath, 'r') as f:
        for line in f:
            tokens.extend(line.strip().split())
    idx2token = list(set(tokens))
    token2idx = {token: idx for idx, token in enumerate(idx2token)}
    return idx2token, token2idx


def main():
    LOGGER.info(f'Start to tokenize train data in {OPT.train_file}')
    idx2token, token2idx = tokenize(OPT.train_file)
    LOGGER.info('Tokenization complete')

    LOGGER.info(f'Start to prepare dataloader. <Train: {OPT.train_file}> <Test: {OPT.test_file}>')
    train_loader, test_loader = load_data(OPT.train_file, OPT.test_file, token2idx)
    LOGGER.info('Dataloader preparation complete')

    LOGGER.info('Start training')
    model = LSTMModel(len(token2idx), 768, 768)
    train(model, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the train file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to the test file')
    parser.add_argument('--pred_file', type=str, default='./pred.txt',
                        help='Path to the output prediction file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Max epochs'),
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Model evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Where to train the model')
    OPT = parser.parse_args()

    LOGGER = get_logger(__name__)

    main()
