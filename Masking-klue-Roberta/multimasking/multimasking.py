from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from konlpy.tag import Mecab
import random
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") # 토크나이저 로드
model = AutoModelWithLMHead.from_pretrained("klue/roberta-large") # 모델 로드

###### Multi-Masking Class ######
class Multi_masking:
    def __init__(self, change_word, input_word, sequence):
        self.change_word = change_word  # 바꿀 현재 단어
        self.will_input_word = input_word   # 바꿀 미래 단어

        self.my_token = tokenizer.encode(self.will_input_word)  # 바꿀 단어 토크나이징
        self.pred_word = self.my_token[1:-1]    # [CLS], [SEP] 제외 리스트

        sequence_list = sequence.split(self.change_word) # 뺄 단어 기준으로 스플릿해서
        self.sequence = f'{sequence_list[0]}{tokenizer.mask_token * len(self.pred_word)}{sequence_list[1]}' # 다시 [MASK] 넣고 재조합

    def split_to_POS(self, sequence):
        POS = mecab.pos(sequence)

        NNG_list = []
        for P in POS:
            if P[1]=='NNG': NNG_list.append(P[0])

        return NNG_list

    def give_masking(self, sequence):
        self.input_ids = tokenizer.encode(sequence, return_tensors="pt")    # 토크나이저로 인코딩
        self.mask_token_index = torch.where(self.input_ids == tokenizer.mask_token_id)[1]   # - torch.where : condition에 따라 값 선택 (if-else같이)
        self.masked_pos = [mask.item() for mask in self.mask_token_index]

    def get_pred(self, index, mask_index):
        token_logits = model(self.input_ids)[0]  # log-odds func
        # mask_token_logits = token_logits[0, mask_token_index, :]
        # mask_token_logits = torch.softmax(mask_token_logits, dim=1)
        mask_token_logits = token_logits.squeeze()  # 차원 다운

        ## topk prediction ##
        mask_hidden_state = mask_token_logits[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=-1)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        scoring = []

        for word in words:
            after_token_id = tokenizer.vocab[word]
            mask_token_log = token_logits[0, self.mask_token_index, :]
            mask_token_log = torch.softmax(mask_token_log, dim=-1)

            scoring.append(mask_token_log[:, after_token_id])

        print(f"Mask {index + 1} Guesses : {words} (Score: {scoring})")     # Top 5 예측

        ## Get the score of pred_word ##
        sought_after_token = tokenizer.decode(self.pred_word[index])
        sought_after_token_id = self.pred_word[index]

        mask_token_logits22 = token_logits[0, self.mask_token_index, :]
        mask_token_logits22 = torch.softmax(mask_token_logits22, dim=1)

        print(f"Score of {sought_after_token}: {mask_token_logits22[0, sought_after_token_id]}")

        # best_guess = ""
        # for j in list_of_list:
        #     best_guess = best_guess + " " + j[0]

        return words[0], mask_token_logits22[0, sought_after_token_id]  # 1등 단어 + 원하는 단어 예측 확률 리턴