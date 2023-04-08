from NGramPredictor import NGramPredictor
import argparse
from tqdm import tqdm

N_GRAM_N = 4
TEST_LINES = 200


def main(train_file=None, test_file=None, pred_file=None, save_model=None, load_model=None):
    ngram_predictor = NGramPredictor(N_GRAM_N)
    if load_model:
        ngram_predictor.load_model(load_model)
    else:
        if train_file:
            # Read words from a file and split by space
            with open(train_file, "r") as file:
                tokenized_dataset = file.read().split()

            print("words count: " + str(len(tokenized_dataset)))

            # Initialize and train the NGramPredictor
            print("Training the NGramPredictor...")
            ngram_predictor.train(tqdm(tokenized_dataset))

            if save_model:
                ngram_predictor.save_model(save_model)
        else:
            raise ValueError("Either provide a train_file or load_model")

    if test_file and pred_file:
        # Read test input from a file and split by lines
        with open(test_file, "r") as test_file:
            test_lines = test_file.readlines()

        # Generate n-gram predictions for each line in the test input
        with open(pred_file, "w") as pred_file:
            print("Generating predictions...")
            correct_cnt = 0
            valid_word_cnt = 0
            for line in tqdm(test_lines[:TEST_LINES]):
                tokens = line.strip().split()
                if len(tokens) <= ngram_predictor.n:
                    continue
                context = tokens[:ngram_predictor.n - 1]
                predictions = []
                for i in range(ngram_predictor.n, len(tokens) - 1):
                    prediction = ngram_predictor.predict(context)
                    if prediction == tokens[i]:
                        correct_cnt += 1
                    valid_word_cnt += 1
                    predictions.append(prediction)
                    context = context[1:] + [tokens[i]]
                pred_file.write(" ".join(predictions) + "\n")
            print("Test Accuracy: {:.2%}".format(correct_cnt / valid_word_cnt))
    elif test_file or pred_file:
        raise ValueError("Both test_file and pred_file must be provided")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", help="Path to the train file", default=None)
    parser.add_argument(
        "--test_file", help="Path to the test file", default=None)
    parser.add_argument(
        "--pred_file", help="Path to the pred file", default=None)
    parser.add_argument(
        "--save_model", help="Path to save the trained model", default=None)
    parser.add_argument(
        "--load_model", help="Path to load a pre-trained model", default=None)
    args = parser.parse_args()
    main(args.train_file, args.test_file, args.pred_file,
         args.save_model, args.load_model)
