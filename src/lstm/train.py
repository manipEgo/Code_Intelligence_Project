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
from tqdm import tqdm


def make_loader(words, token2idx, idx2token):
    dataset = CodeDataset(OPT.seq_length, words, token2idx, idx2token)
    return DataLoader(dataset, batch_size=OPT.batch_size, shuffle=True)


def train(model: LSTMModel,
          train_loader: DataLoader,
          test_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer):
    for epoch in range(OPT.max_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in tqdm(train_loader, desc='Train'):
            h0, c0 = model.init_hidden(inputs.size(0))
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()

            outputs, _ = model(inputs, (h0, c0))
            loss = criterion(outputs.transpose(1, 2), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if epoch % OPT.eval_freq == 0:
                last_digit = outputs[0][-1]
                softmax = torch.softmax(last_digit, dim=0)
                _, predicted = torch.max(softmax, 0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if epoch % OPT.eval_freq == 0:
            accuracy, loss = evaluate(model, criterion, test_loader)
            print(f'Epoch {epoch}\n'
                  f'Train Accuracy: {correct / total:.3f}\tTrain Loss: {running_loss / total:.3f}\t'
                  f'Test Accuracy: {accuracy:.3f}\tTest Loss: {loss:.3f}')


@torch.no_grad()
def evaluate(model: LSTMModel, criterion: nn.Module, loader: DataLoader):
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc='eval'):
        h0, c0 = model.init_hidden(inputs.size(0))
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs, _ = model(inputs, (h0, c0))
        loss = criterion(outputs.transpose(1, 2), labels)
        running_loss += loss.item()

        last_digit = outputs[0][-1]
        softmax = torch.softmax(last_digit, dim=0)
        _, predicted = torch.max(softmax, 0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total, running_loss / total


def tokenize(filepath: str, sample_size: int):
    words = []
    with open(filepath, 'r') as f:
        index = 0
        for line in f:
            if index >= sample_size:
                break
            words.extend(line.strip().split())
            index += 1
    unique_tokens = Counter(words)
    unique_tokens.update(['<unknown/>'])
    unique_tokens = sorted(unique_tokens, key=unique_tokens.get, reverse=True)
    idx2token = list(unique_tokens)
    token2idx = {token: idx for idx, token in enumerate(idx2token)}
    return words, idx2token, token2idx


def main():
    LOGGER.info(f'Start to tokenize train data in {OPT.train_file}')
    train_words, train_idx2token, train_token2idx = tokenize(OPT.train_file, OPT.sample_size)
    test_words, test_idx2token, test_toekn2idx = tokenize(OPT.test_file, int(OPT.sample_size // 4))
    LOGGER.info('Tokenization complete')

    LOGGER.info(f'Start to prepare dataloader. <Train: {OPT.train_file}> <Test: {OPT.test_file}>')
    train_loader = make_loader(train_words, train_token2idx, train_idx2token)
    test_loader = make_loader(test_words, train_token2idx, train_idx2token)
    # train_loader, test_loader = load_data(train_words, train_token2idx, train_idx2token)
    LOGGER.info('Dataloader preparation complete')

    LOGGER.info('Start training')
    if OPT.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = OPT.device

    try:
        model = LSTMModel(device, len(train_token2idx), 32, 32).to(device)
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs'),
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Model evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Where to train the model')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size of train file')
    OPT = parser.parse_args()

    LOGGER = get_logger(__name__)

    main()
