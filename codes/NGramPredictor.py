from collections import defaultdict, Counter
import pickle
import os


class NGramPredictor:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)

    def _generate_ngrams(self, tokens):
        return zip(*[tokens[i:] for i in range(self.n)])

    def train(self, tokenized_dataset):
        for tokens in tokenized_dataset:
            for ngram in self._generate_ngrams(tokens):
                context, token = tuple(ngram[:-1]), ngram[-1]
                self.model[context][token] += 1

    def predict(self, context):
        if len(context) != self.n - 1:
            raise ValueError("Context length must be equal to n-1")
        context = tuple(context)
        if context in self.model:
            return self.model[context].most_common(1)[0][0]
        else:
            return None

    def save_model(self, file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)
