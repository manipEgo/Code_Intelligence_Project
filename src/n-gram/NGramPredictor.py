from collections import defaultdict, Counter
import pickle
import os


class NGramPredictor:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, tokenized_dataset):
        context = []
        for token in tokenized_dataset:
            if len(context) < self.n - 1:
                context.append(token)
            else:
                self.model[tuple(context)][token] += 1
                context = context[1:] + [token]

    def predict(self, context):
        if len(context) != self.n - 1:
            raise ValueError("Context length must be equal to n-1")
        context = tuple(context)
        if context in self.model:
            return self.model[context].most_common(1)[0][0]
        else:
            return "DISTINCT_NULL"

    def predict_list(self, context, top_n=100):
        if len(context) != self.n - 1:
            raise ValueError("Context length must be equal to n-1")
        context = tuple(context)
        if context in self.model:
            total_count = sum(self.model[context].values())
            return [(word, count / total_count) for word, count in self.model[context].most_common(top_n)]
        else:
            return [("DISTINCT_NULL", 1)]

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
