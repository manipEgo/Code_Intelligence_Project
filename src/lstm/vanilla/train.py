import argparse
import cmd
import matplotlib.pyplot as plt
import pickle
import traceback
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from dataset import CodeDataset
from model import VanillaLSTM
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm


def make_loader(words: List[str], token2idx: Dict[str, int], idx2token: List[str]):
    dataset = CodeDataset(OPT.seq_length, words, token2idx, idx2token)
    return DataLoader(dataset, batch_size=OPT.batch_size, shuffle=True)


def train(model: VanillaLSTM,
          train_loader: DataLoader,
          test_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer):
    x_axis, train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], [], []
    for epoch in range(OPT.max_epochs):
        print(f'Epoch {epoch}')
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in tqdm(train_loader, desc='Train'):
            h0, c0 = model.init_hidden(inputs.size(0))
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()

            outputs, _ = model(inputs, (h0, c0))
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

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
            x_axis.append(epoch)
            train_accuracy.append(correct / total)
            train_loss.append(running_loss / len(train_loader))
            accuracy, loss = evaluate(model, criterion, test_loader)
            test_accuracy.append(accuracy)
            test_loss.append(loss)
            print(f'Train Accuracy: {train_accuracy[-1]:.3f}\tTrain Loss: {train_loss[-1]:.3f}\t'
                  f'Test Accuracy: {test_accuracy[-1]:.3f}\tTest Loss: {test_loss[-1]:.3f}')

    torch.save(model.state_dict(), OPT.save_path)

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_accuracy, label='train accuracy')
    ax.plot(x_axis, train_loss, label='train loss')
    ax.plot(x_axis, test_accuracy, label='test accuracy')
    ax.plot(x_axis, test_loss, label='test loss')
    ax.legend()
    fig.savefig('./Accuracu-Loss-Curve.png')


@torch.no_grad()
def evaluate(model: VanillaLSTM, criterion: nn.Module, loader: DataLoader):
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc='eval'):
        h0, c0 = model.init_hidden(inputs.size(0))
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs, _ = model(inputs, (h0, c0))
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
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
    print(f'Start to tokenize train data in {OPT.train_file}')
    train_words, train_idx2token, train_token2idx = tokenize(OPT.train_file, OPT.sample_size)
    test_words, test_idx2token, test_token2idx = tokenize(OPT.test_file, OPT.test_lines)
    print('Tokenization complete')

    if OPT.save_dict:
        print('Start to save dict')
        with open(OPT.save_dict, 'wb') as f:
            pickle.dump({
                'words': train_words,
                'idx2token': train_idx2token,
                'token2idx': train_token2idx
            }, file=f)
        print('Dict save complete')

    print(f'Start to prepare dataloader. <Train: {OPT.train_file}> <Test: {OPT.test_file}>')
    train_loader = make_loader(train_words, train_token2idx, train_idx2token)
    test_loader = make_loader(test_words, train_token2idx, train_idx2token)
    print('Dataloader preparation complete')

    print('Start training')
    try:
        model = VanillaLSTM(DEVICE, len(train_token2idx), 32, 32).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=OPT.learning_rate)
        with open('checkpoint/args.txt', 'a') as f:
            print(model, file=f)
            print(criterion, file=f)
            print(optimizer, file=f)

        print(f'Train model on {DEVICE}')
        train(model, train_loader, test_loader, criterion, optimizer)
    except torch.cuda.OutOfMemoryError:
        print(torch.cuda.memory_summary(device=DEVICE))
        traceback.print_exc()
        
        
class CLI(cmd.Cmd):
    prompt = 'Vanilla LSTM >>> '
    intro = "Welcome! Type 'help' to list commands"

    def __init__(self, model: VanillaLSTM, token2idx: Dict[str, int], idx2token: List[str]):
        super().__init__()
        self.model = model
        self.token2idx = token2idx
        self.idx2token = idx2token

    def default(self, arg: str):
        try:
            inputs = torch.LongTensor([self.token2idx[token] for token in arg.split()]).to(DEVICE)
        except KeyError:
            print(f'Contain unknown token: {arg}')
            return
        inputs = inputs.unsqueeze(0)
        h0, c0 = self.model.init_hidden(inputs.size(0))
        outputs, _ = self.model(inputs, (h0, c0))
        last_token = outputs[0][-1]
        softmax = torch.softmax(last_token, dim=0)
        _, predicted = torch.max(softmax, 0)
        
        print(f'Next token: {self.idx2token[predicted]}')

    def do_exit(self, args):
        return True


def cli_main():
    with open(OPT.save_dict, 'rb') as f:
        save_dict = pickle.load(f)
        token2idx = save_dict['token2idx']
        idx2token = save_dict['idx2token']
    model = VanillaLSTM(DEVICE, len(token2idx), 32, 32).to(DEVICE)
    model.load_state_dict(torch.load(OPT.save_path))

    cli = CLI(model, token2idx, idx2token)
    cli.cmdloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        help='Path to the train file')
    parser.add_argument('--test_file', type=str,
                        help='Path to the test file')
    parser.add_argument('--pred_file', type=str, default='./pred.txt',
                        help='Path to the output prediction file'),
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length needed to predict')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Max epochs'),
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Model evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Where to train the model')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size of train file')
    parser.add_argument('--test_lines', type=int, default=500,
                        help='Sample size of test file')
    parser.add_argument('--save_path', type=str, default='./save.pth',
                        help='Path to save model')
    parser.add_argument('--save_dict', type=str, default=None,
                        help='Path to save the words dict (words, idx2token, token2idx)')
    parser.add_argument('--interactive', action='store_true', default=False)
    OPT = parser.parse_args()

    DEVICE = None
    if OPT.device is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = OPT.device

    if OPT.interactive:
        cli_main()
    else:
        with open('./checkpoint/args.txt', 'w') as f:
            print(OPT, file=f)
        main()
