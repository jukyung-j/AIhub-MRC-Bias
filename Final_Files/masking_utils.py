from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch

"""
노후 어린이집 누수 개보수에 책정된 경비가 얼마야
0 ['어린이집'] ['유치원', '유아원', '놀이방'] 원 [['엄', '화', '미래상', '제갈', '홍', '판상', '싫음', '형', '천', '왕']]

얼마의 비용으로 노후 어린이집 누수 문제를 해결하니
1 ['해결'] ['양자택일', '득점', '해소', '예비음', '위치결정', '잡도리', '성패', '대승리', '히트', '예정'] 원 [['엄', '화', '미래상', '제갈', '홍', '판상', '싫음', '형', '천', '왕']]
"""

def replace_tokens_for_masking(question, text_to_replaced, tokenizer):
    tokens_to_replaced = tokenizer.tokenize(text_to_replaced)
    query_tokens = tokenizer.tokenize(question)

    tokens = []

    i = 0
    while True:
        if i >= len(query_tokens):
            break

        if i + len(tokens_to_replaced) >= len(query_tokens):
            tokens.append(query_tokens[i])
            i += 1
            continue

        check_token_exact = True
        for t in range(len(tokens_to_replaced)):
            if query_tokens[i + t].replace('##', '') != tokens_to_replaced[t].replace('##', ''):
                check_token_exact = False

        if check_token_exact is True:
            tokens.append('[MASK]')
            i += len(tokens_to_replaced)
            continue

        tokens.append(query_tokens[i])
        i += 1

    try:
        index = tokens.index('[MASK]')
    except:
        question = question.replace(text_to_replaced, ' [MASK] ')
        tokens = tokenizer.tokenize(question)

    return tokens


def make_mask_input(input_tokens, siblings, tokenizer):
    #print(input_tokens)
    mask_index = input_tokens.index('[MASK]')

    results = []

    for s, word in enumerate(siblings):
        answer_tokens = tokenizer.tokenize(word)

        for w in range(len(answer_tokens)):
            mask_answer_tokens = answer_tokens.copy()
            token_ids = [mask_answer_tokens[w]]
            token_ids = tokenizer.convert_tokens_to_ids(tokens=token_ids)
            token_ids = token_ids[0]

            mask_answer_tokens[w] = '[MASK]'

            tokens = []
            tokens.extend(input_tokens[0: mask_index])
            tokens.extend(mask_answer_tokens)
            tokens.extend(input_tokens[mask_index + 1: len(input_tokens)])

            results.append((tokens, s, token_ids))

    return results


def get_sibling_prob(token_inputs, tokenizers, siblings, model, device, max_length=32):
    input_ids = np.zeros(shape=[len(token_inputs), max_length], dtype=np.int32)
    attention_ids = np.zeros(shape=[len(token_inputs), max_length], dtype=np.int32)

    sibling_counts = [0] * len(siblings)
    sibling_indexes = []
    decoding_indexes = []
    mask_indexes = []
    #print(token_inputs)
    for t, token_input in enumerate(token_inputs):
        #print(token_input)

        tokens, sibling_index, decoding_index = token_input
        mask_index = tokens.index('[MASK]')
        mask_indexes.append(mask_index)

        sibling_counts[sibling_index] += 1
        sibling_indexes.append(sibling_index)

        decoding_indexes.append(decoding_index)

        ids = tokenizers.convert_tokens_to_ids(tokens=tokens)

        length = len(ids)
        if length > max_length:
            length = max_length

        input_ids[t, 0: length] = ids[0: length]
        attention_ids[t, 0: length] = 1
    #print(np.max(input_ids))
    input_ids = torch.from_numpy(input_ids)
    input_ids = input_ids.to(device)
    attention_ids = torch.from_numpy(attention_ids)
    attention_ids = attention_ids.to(device)

    with torch.no_grad():
        #print(input_ids.shape)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_ids
        ).logits
        logits = torch.softmax(logits, dim=0)
        logits = logits.detach().cpu().numpy()

    sibling_prob = [0] * len(siblings)
    try:
        for t, _ in enumerate(token_inputs):
            sibling_index = sibling_indexes[t]
            decoding_index = decoding_indexes[t]
            mask_index = mask_indexes[t]

            sibling_prob[sibling_index] += logits[t, mask_index, decoding_index]
    except:
        return get_sibling_prob(token_inputs, tokenizers, siblings, model, device=device, max_length=max_length * 2)

    for p in range(len(sibling_prob)):
        sibling_prob[p] = sibling_prob[p] / sibling_counts[p]
    sibling_prob = np.array(sibling_prob)
    #print(sibling_prob)
    sib_idx = sibling_prob.argmax()

    return sib_idx


def get_sibling_prediction(question, text_to_replaced, siblings, tokenizer, model, device):
    #print('check', question, ',', text_to_replaced)
    input_tokens = replace_tokens_for_masking(question=question,
                                              text_to_replaced=text_to_replaced,
                                              tokenizer=tokenizer)

    results = make_mask_input(input_tokens=input_tokens,
                              siblings=siblings,
                              tokenizer=tokenizer)

    # token_inputs, tokenizers, siblings, model
    sib_idx = get_sibling_prob(
        token_inputs=results,
        tokenizers=tokenizer,
        siblings=siblings,
        model=model,
        device=device
    )

    return siblings[sib_idx]


if __name__ == '__main__':
    #노후 어린이집 누수 개보수에 책정된 경비가 얼마야
    #0 ['어린이집']
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
    model = AutoModelForMaskedLM.from_pretrained("klue/roberta-base")

    result_file = open('행정' + '_result_file.txt', 'w', encoding='utf-8')

    question = '노후 어린이집 누수 개보수에 책정된 경비가 얼마야'
    text_to_replaced = '어린이집'
    siblings = ['유치원', '유아원', '놀이방']

    input_tokens = replace_tokens_for_masking(question='노후 어린이집 누수 개보수에 책정된 경비가 얼마야',
                                              text_to_replaced='어린이집',
                                              tokenizer=tokenizer)
    results = make_mask_input(input_tokens=input_tokens,
                              siblings=siblings,
                              tokenizer=tokenizer)

    for result in results:
        print(result)

    # token_inputs, tokenizers, siblings, model
    sib_idx = get_sibling_prob(
        token_inputs=results,
        tokenizers=tokenizer,
        siblings=siblings,
        model=model
    )

    print(siblings[sib_idx])

