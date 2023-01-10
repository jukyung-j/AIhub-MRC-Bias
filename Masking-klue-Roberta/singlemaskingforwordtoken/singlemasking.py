from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from konlpy.tag import Mecab
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") # 토크나이저 로드
model = AutoModelWithLMHead.from_pretrained("klue/roberta-large") # 모델 로드

###### Singe-Masking for Multi-Masking Class ######
class Single_masking:
    def __init__(self, change_word, input_word, sequence):
        self.change_word = change_word # 바꿀 현재 단어
        self.will_input_word = input_word # 바꿀 미래 단어

        self.my_token = tokenizer.encode(self.will_input_word)  # 바꿀 단어 토크나이징
        self.pred_word = self.my_token[1:-1]    # [CLS], [SEP] 제외 리스트

        decode_word = ''
        for d in self.pred_word:
            decode_word = decode_word+tokenizer.decode(d)
        sequence = sequence.replace(change_word, decode_word)

        self.sequence_list = []
        for d in self.pred_word:
            self.sequence_list.append((sequence.replace(tokenizer.decode(d), f'{tokenizer.mask_token}')).replace('##', ''))

    def split_to_POS(self, sequence):
        POS = mecab.pos(sequence)

        NNG_list = []
        for P in POS:
            if P[1]=='NNG': NNG_list.append(P[0])

        return NNG_list

    def give_masking(self, sequence):
        input_ids = tokenizer.encode(sequence, return_tensors="pt")  # 토크나이저로 인코딩
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]  # - torch.where : condition에 따라 값 선택 (if-else같이)
        masked_pos = [mask.item() for mask in mask_token_index]

        return input_ids, mask_token_index, masked_pos

    def get_pred(self, index, input_ids, mask_token_index, masked_pos):
        token_logits = model(input_ids)[0]  # log-odds func
        mask_token_logits = token_logits.squeeze() # 차원 다운

        ## 1. topk prediction ##
        mask_hidden_state = mask_token_logits[masked_pos]
        idx = torch.topk(mask_hidden_state, k=5, dim=-1)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        scoring = []

        for word in words:
            after_token_id = tokenizer.vocab[word]
            mask_token_log = token_logits[0, mask_token_index, :]
            mask_token_log = torch.softmax(mask_token_log, dim=-1)

            scoring.append(mask_token_log[:, after_token_id])

        print(f"Mask {index + 1} Guesses : {words} (Score: {scoring})")

        ## 2. Get the score of pred_word ##
        sought_after_token = tokenizer.decode(self.pred_word[index])
        sought_after_token_id = self.pred_word[index]

        mask_token_logits2 = token_logits[0, mask_token_index, :]
        mask_token_logits2 = torch.softmax(mask_token_logits2, dim=-1)

        print(f"Score of {sought_after_token}: {mask_token_logits2[:, sought_after_token_id]}")

        return words[0], mask_token_logits2[:, sought_after_token_id]  # 1등 단어 + 원하는 단어 예측 확률 리턴