import argparse
import traceback
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from dataset import CodeDataset
from logger import get_logger
from model import LSTMModel
from torch.utils.data import DataLoader


def load_data(token2idx):
    train_dataset = CodeDataset(OPT.seq_length, token2idx)
    test_dataset = CodeDataset(OPT.seq_length, token2idx)
    train_loader = DataLoader(train_dataset, batch_size=OPT.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=OPT.batch_size, shuffle=True)
    return train_loader, test_loader


def train(model: LSTMModel,
          train_loader: DataLoader,
          test_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer):
    for epoch in range(OPT.max_epochs):
        running_loss = 0.0
        h0, c0 = model.init_hidden(OPT.batch_size)
        for inputs, labels in train_loader:
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()

            outputs, _ = model(inputs, (h0, c0))
            loss = criterion(outputs.transpose(1, 2), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % OPT.eval_freq == 0:
            accuracy, loss = evaluate(model, criterion, test_loader)
            print(f'Epoch {epoch}: accuracy {accuracy}, loss {loss}')


@torch.no_grad()
def evaluate(model: LSTMModel, criterion: nn.Module, loader: DataLoader):
    running_loss = 0.0
    correct, total = 0, 0
    h0, c0 = model.init_hidden(OPT.batch_size)
    for inputs, labels in loader:
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs, _ = model(inputs, (h0, c0))
        loss = criterion(outputs.transpose(1, 2), labels)
        running_loss += loss.item()

        last_digit = outputs[0][-1]
        _, predicted = torch.max(torch.softmax(last_digit, dim=1))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total, running_loss / len(loader)


def tokenize(filepath):
    tokens = []
    with open(filepath, 'r') as f:
        for line in f:
            tokens.extend(line.strip().split())
    unique_tokens = Counter(tokens)
    unique_tokens = sorted(unique_tokens, key=unique_tokens.get, reverse=True)
    idx2token = list(unique_tokens)
    token2idx = {token: idx for idx, token in enumerate(idx2token)}
    return idx2token, token2idx


def main():
    LOGGER.info(f'Start to tokenize train data in {OPT.train_file}')
    idx2token, token2idx = tokenize(OPT.train_file)
    LOGGER.info('Tokenization complete')

    LOGGER.info(f'Start to prepare dataloader. <Train: {OPT.train_file}> <Test: {OPT.test_file}>')
    train_loader, test_loader = load_data(token2idx)
    LOGGER.info('Dataloader preparation complete')

    LOGGER.info('Start training')
    if OPT.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = OPT.device

    try:
        model = LSTMModel(device, len(token2idx), 32, 32).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=OPT.learning_rate)
        train(model, train_loader, test_loader, criterion, optimizer)
    except torch.cuda.OutOfMemoryError:
        print(torch.cuda.memory_summary(device=device))
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the train file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to the test file')
    parser.add_argument('--pred_file', type=str, default='./pred.txt',
                        help='Path to the output prediction file'),
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length needed to predict')
    parser.add_argument('--batch_size', type=int, default=8,
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
