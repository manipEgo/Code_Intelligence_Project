from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
import json
import os


class NGramPredictor:
    def __init__(self, n, pretrained, lit_file):
        self.n = n
        special_tokens = self.get_special_tokens(lit_file)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_special_tokens(self, path):
        lits = json.load(open(path))
        tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
        for lit in lits["str"]:
            tokens.append(f"<STR_LIT:{lit}>")
        for lit in lits["num"]:
            tokens.append(f"<NUM_LIT:{lit}>")
        for lit in lits["char"]:
            tokens.append(f"<CHAR_LIT:{lit}>")
        return tokens

    def predict_list(self, context, top_n=100):
        self.model.eval()
        inputs = self.tokenizer(context, return_tensors="tf")
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        scores = torch.softmax(logits[0][-1]).detach().numpy()
        idxs = np.argsort(scores)
        result = []
        for i in range(top_n):
            result.append((self.tokenizer.convert_ids_to_tokens(idxs[i], scores[i])))
        return result
        # self.model.eval()

        # total_pred = []
        # total_gt = []

        # for step, batch in enumerate(eval_dataloader):
        #     inputs = batch.to(device)

        #     with torch.no_grad():
        #         outputs = self.model(inputs)
        #         pred_scores = outputs[0]
        #         pred_ids = pred_scores.argmax(-1)

        #     all_pred = []
        #     all_gt = []
        #     prev_pred = None
        #     for pred, gt in zip(pred_ids, inputs):
        #         pred = pred.cpu().tolist()
        #         gt = gt.cpu().tolist()

        #         for i, y in enumerate(gt):
        #             if i == 0:
        #                 if y in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
        #                     now_gt = [y]
        #                     now_pred = [0] if prev_pred is None else [prev_pred]
        #                     all_pred.append(self.DecodeIds(now_pred).strip().split()[0])
        #                     all_gt.append(self.DecodeIds(now_gt).strip())
        #                     now_gt = []
        #                     now_pred = []
        #                 else:
        #                     now_gt = [y]
        #                     now_pred = [0] if prev_pred is None else [prev_pred]
        #             else:
        #                 if self.tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
        #                     if len(now_gt) > 0:
        #                         try:
        #                             all_pred.append(self.DecodeIds(now_pred).strip().split()[0])
        #                         except IndexError:
        #                             all_pred.append("<SPACE>")
        #                         all_gt.append(self.DecodeIds(now_gt).strip())
        #                         now_gt = []
        #                         now_pred = []
        #                 if y in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id] or self.tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT"):
        #                     if len(now_gt) > 0:
        #                         try:
        #                             all_pred.append(self.DecodeIds(now_pred).strip().split()[0])
        #                         except IndexError:
        #                             all_pred.append("<SPACE>")
        #                         all_gt.append(self.DecodeIds(now_gt).strip())
        #                     now_gt = [y]
        #                     now_pred = [pred[i-1]]
        #                     try:
        #                         all_pred.append(self.DecodeIds(now_pred).strip().split()[0])
        #                     except IndexError:
        #                         all_pred.append("<SPACE>")
        #                     all_gt.append(self.DecodeIds(now_gt).strip())
        #                     now_gt = []
        #                     now_pred = []
        #                     continue
        #                 now_gt.append(y)
        #                 now_pred.append(pred[i-1])
        #     assert len(all_pred) == len(all_gt)

        #     total_pred.extend(all_pred)
        #     total_gt.extend(all_gt)
