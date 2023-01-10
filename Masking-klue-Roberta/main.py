from multimasking import multimasking
from singlemaskingforwordtoken import singlemasking
from transformers import AutoModelWithLMHead, AutoTokenizer

# 1. 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") # 토크나이저 로드
model = AutoModelWithLMHead.from_pretrained("klue/roberta-large") # 모델 로드

# *변경될 형제어따라 조사 변경
def has_conda(word: str):  # 아스키(ASCII) 코드 공식에 따라 입력된 단어의 마지막 글자 받침 유무를 판단해서 뒤에 붙는 조사를 리턴하는 함수
    last = word[-1]  # 입력된 word의 마지막 글자를 선택해서
    criteria = (ord(last) - 44032) % 28  # 아스키(ASCII) 코드 공식에 따라 계산 (계산법은 다음 포스팅을 참고하였습니다 : http://gpgstudy.com/forum/viewtopic.php?p=45059#p45059)
    if criteria == 0:  # 종성없음
        return False
    else:  # 종성있음
        return True
def change_conda(sequence, chagne_word, sibling):
    if has_conda(sibling):
        operation_text = sequence.replace(f'{chagne_word}는', f'{chagne_word}은')
        operation_text = sequence.replace(f'{chagne_word}를', f'{chagne_word}을')
        operation_text = sequence.replace(f'{chagne_word}가', f'{chagne_word}이')
        operation_text = sequence.replace(f'{chagne_word}와', f'{chagne_word}과')
        operation_text = sequence.replace(f'{chagne_word}랑', f'{chagne_word}이랑')
    else:
        operation_text = sequence.replace(f'{chagne_word}은', f'{chagne_word}는')
        operation_text = sequence.replace(f'{chagne_word}를', f'{chagne_word}을')
        operation_text = sequence.replace(f'{chagne_word}이', f'{chagne_word}가')
        operation_text = sequence.replace(f'{chagne_word}과', f'{chagne_word}와')
        operation_text = sequence.replace(f'{chagne_word}이랑', f'{chagne_word}랑')
    return operation_text


# 2. 함수 작동
############## Main에 바꿔넣기 귀찮아서 2가지 함수로 빼놓음 ################

######################## For Multi-Masking! ##########################
def One(sequence, change_word, will_input_word):
    scores = []  # 각 토큰 확률 저장할거
    multi_masking = multimasking.Multi_masking(change_word, will_input_word, sequence)  # 마스킬 클래스 선언하고
    # print(multi_masking.split_to_POS(sequence))

    for index in range(len(multi_masking.pred_word)):  # 토큰 갯수 만큼 돌릴거임
        multi_masking.give_masking(multi_masking.sequence)  # [MASK] 맥이고
        guess, score = multi_masking.get_pred(index, multi_masking.masked_pos[0])  # 예측 단어랑 스코어 prediction하고
        scores.append(score)

        for s in range(len(multi_masking.sequence)):  # 다음 단어에 영향주려고 [MASK] 원하는 단어로 바꿀거임
            if multi_masking.sequence[s] == '[':
                multi_masking.sequence = multi_masking.sequence[:s] + tokenizer.decode(
                    multi_masking.pred_word[index]) + multi_masking.sequence[s + 6:]
                multi_masking.sequence = multi_masking.sequence.replace('##', '')
                break
        print(f'{index}: ', multi_masking.sequence)
        print('----------------------------------------------------------------\n')

    avg_score = sum(scores) / len(scores)
    print(f'Average Prob: {avg_score}')  # 모든 토큰 했을 때 확률 평균임

    return avg_score
######################################################################

######################## For Single-Masking! ##########################
def Two(sequence, change_word, will_input_word):
    scores = []  # 각 토큰 확률 저장할거
    sequence = change_conda(sequence, change_word, will_input_word)
    single_masking = singlemasking.Single_masking(change_word, will_input_word, sequence)

    for index, seq in enumerate(single_masking.sequence_list):
        input_ids, mask_token_index, masked_pos = single_masking.give_masking(seq)
        word, prob = single_masking.get_pred(index, input_ids, mask_token_index, masked_pos[0])
        scores.append(prob)

        print(f'{index + 1} senquence:', seq)
        print(f'Top word: {word}, score:  {prob}')
        print('----------------------------------------------------------------')

    avg_score = sum(scores)/len(scores)
    print(f'Average Prob: {avg_score}')

    return avg_score
######################################################################

if "__main__" == __name__:
    sequence_list = ['나는 아침에 훈제오리로스랑 계란후라이를 먹어서 너무 기부니가 좋다.']
    change_word_list = [['훈제오리로스']]
    korlex_list = [['매운진라면', '상한쌀밥', '똥무더기']]

    One_Scores, Two_Scores = [], [] # dict으로 바꾸고 싶은데 만다 안되누ㅠ
    top_sibling_word, top_score = '', 0

    # # Run One & Two
    # for korlex in korlex_list:
    #     print('Print One ============================================')
    #     One_Scores.append([korlex, One('탄탄멘', korlex)])
    #     # One_Scores = sorted(One_Scores.items(), reverse=True) # dict
    #     One_Scores.sort(key=lambda x: x[1], reverse=True) # list
    #
    #     print('\nPrint Two ============================================')
    #     Two_Scores.append([korlex, Two('탄탄멘', korlex)])
    #     # Two_Scores = sorted(Two_Scores.items(), reverse=True)
    #     Two_Scores.sort(key=lambda x: x[1], reverse=True)

    for seq, word, korlex in zip(sequence_list, change_word_list, korlex_list):
        for sibling in korlex:
            print(f'\nPrint {sibling} ============================================\n')
            Two_Scores.append(Two(seq, word[0], sibling))
            cur_score = float(str(Two_Scores[-1]).split('[')[1].split(']')[0])
            if top_score < cur_score:
                top_score = cur_score
                top_sibling_word = sibling

    print(top_sibling_word)

    # print('Print Average of One ==================================')
    # print(One_Scores)
    # print('\nPrint Average of Two ==================================')
    # print(Two_Scores)