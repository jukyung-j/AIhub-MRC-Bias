import multiprocessing
import itertools
from itertools import product

import glob
import os
import re
import json
import random
from konlpy.tag import Mecab
import krx_api
from pyjosa.josa import Josa
import masking
from datetime import datetime
from multiprocessing import Pool, Process


class Augmentation:
    def __init__(self, file_format):
        self.mecab = Mecab()
        self.labeled_train_files, self.labeled_valid_files, self.origin_train_files, self.origin_valid_files = [], [], [], []
        for name in glob.glob(file_format, recursive=True):  # 데이터 파일 가져오기
            self.labeled_train_files.append(name)
        #     if '라벨링데이터' in name:
        #         if 'Training' in name:
        #             self.labeled_train_files.append(name)
        #         else:
        #             self.labeled_valid_files.append(name)
        #     elif '원천데이터' in name:
        #         if 'Training' in name:
        #             self.origin_train_files.append(name)
        #         else:
        #             self.origin_valid_files.append(name)
        #     else:
        #         if 'Training' in name:
        #             self.labeled_train_files.append(name)
        #         else:
        #             self.labeled_valid_files.append(name)
        ## Load Korlex Files
        ssInfo_path = "./dic/korlex_ssInfo.pkl"
        seIdx_path = "./dic/korlex_seIdx.pkl"
        reIdx_path = "./dic/korlex_reIdx.pkl"
        self.krx_json_api = krx_api.KorLexAPI(ssInfo_path=ssInfo_path,
                                 seIdx_path=seIdx_path,
                                 reIdx_path=reIdx_path)
        self.krx_json_api.load_synset_data()
        self.krx = {}

    def extract_sibling(self, word):    # Korlex 형제어 뽑기
        word_list, replace_list = [], []

        if word not in self.krx:    # 형제어 dict에 없으면 추가
            sibling_words = self.krx_json_api.search_word(word=word, ontology=krx_api.ONTOLOGY.KORLEX.value)

            # Extract Siblings
            for t_s in sibling_words:
                sub_siblings = str(t_s[0]).split('siblings=')[1:]

                for sub in sub_siblings:
                    sib_list = sub.split('KorLexResult(')[0].split('text=')

                    for sib in sib_list:
                        m = re.search('\'(.+?)\', sense', sib)
                        if m:
                            found = m.group(1)
                            word_list.append(found)

            word_list = krx_api.extract_en(word_list)

            # 원시 단어 중복 제거
            word_list = [w for w in word_list if w not in word]

            # 중복 단어 제거
            set_word = set(word_list)
            word_list = list(set_word)
            self.krx[word] = word_list
        else:
            word_list = self.krx[word]

        if len(word_list) > 30: # 10개만 추출
            word_list = random.sample(word_list, 30)
        return word_list

    def find_noun(self, sentence):  # 명사 찾기
        is_noun = False
        word, nouns = "", []
        # pattern 제외
        pattern = ["누가","누구","누굴","누군지","언제","어디","어딜","며칠","며칟날","무슨","무엇","무얼","뭐","뭘","뭔","얼마나","얼마","얼만큼","어떤","어느","왜","어쨰서","어떻게","어떻","어떠","어땠","어때","어떨","몇","어찌하여"]

        is_m = False

        pos_sentence = self.mecab.pos(sentence)
        sentence = sentence.split()
        index = 0
        pos_word = ""
        is_continue = False

        for w, p in pos_sentence:   # 띄워쓰기 기준으로 복합명사
            if is_continue:
                pos_word += w
            else:
                pos_word = w

            if 'NNG' in p and w not in pattern and not is_m:
                word += w

            else:
                if word not in nouns and len(word) > 0:
                    nouns.append(word)
                word = ""

            if w == '몇':
                is_m = True
            else:
                is_m = False

            if sentence[index] == pos_word:
                index += 1
                is_noun, is_continue = False, False
                if word not in nouns and len(word) > 0:
                    nouns.append(word)
                word, pos_word = "", ""

            else:
                is_continue = True


        # Korlex 없는 단어 제외
        words = nouns[:]
        sibling_list = []
        for i, word in enumerate(nouns):
            sibling_list.append(self.extract_sibling(word))

            if not sibling_list[i]: # 형제어 없으면 제외
                words.remove(word)

        if len(words) > 1:
            choice_nouns = random.sample(words, 2)
            sibling = []

            for choice_noun in choice_nouns:
                index = nouns.index(choice_noun)
                sibling.append(sibling_list[index])

            choice_noun = choice_nouns
        elif len(words) == 1:
            choice_noun = random.sample(words, 1)
            index = nouns.index(choice_noun[0])
            sibling = sibling_list[index]

            choice_noun = [choice_noun[0], '']
            sibling = [sibling, []]
        else:
            choice_noun = ['', '']
            sibling = [[], []]

        return choice_noun, sibling

    def test_mask(self, question, origin_word, sibling_list):
        word = [0 for _ in range(len(sibling_list))]

        for i in range(len(sibling_list)):
            word[i] = sibling_list[i][0]

        return word

    def transfer_sentence(self, sentences, answers):
        # 평서문 바꾸기
        mecab = Mecab()
        pattern = "누가|누구|누굴|누군지|언제|어디|어딜|며칠|며칟날|무슨|무엇|무얼|뭐|뭘|뭔|얼마나|얼마|얼만큼|어떤|어느|왜|어쨰서|어떻게|어떻|어떠|어땠|어때|어떨|몇|어찌하여"
        josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', ' JKQ', 'JX', ' JC']
        sign = ['SF']   # 질문에서 끝 문자 ?,! 등등... 제외
        temp = pattern[:]
        sentence_list = []
        choice_answer, answer_sibling = [], []
        is_exist = True
        jos_pattern = ''

        for j in range(len(answers)):
            # answer 형제어 바꾸기
            origin = answers[j]
            noun_answer = self.mecab.nouns(answers[j])
            # Korlex 없는 단어 제외
            words = noun_answer[:]
            sibling_list = []
            for i, word in enumerate(noun_answer):
                if word not in self.krx:
                    sibling_list.append(self.extract_sibling(word))
                    self.krx[word] = sibling_list[i]
                    if not sibling_list[i]:
                        words.remove(word)
                else:
                    if len(self.krx[word]) > 10:
                        sibling_list.append(random.sample(self.krx[word], 10))
                    else:
                        sibling_list.append(self.krx[word])

                    if not sibling_list[i]:
                        words.remove(word)

            if len(words) > 0:
                choice_answer.append(random.sample(words, 1))
                index = noun_answer.index(choice_answer[j][0])
                answer_sibling.append(sibling_list[index])

            else:
                is_exist = False

        is_qus = False #패턴에 걸리는 아이 확인
        for j, sentence in enumerate(sentences):
            pattern = temp[:]
            s = mecab.pos(sentence)
            is_josa, is_m = False, False
            is_qus = False

            # delete sign
            for w, p in s:
                if p in sign:
                    sentence = sentence.replace(w, "")
                    s = mecab.pos(sentence)

            for i in range(len(s) - 1):
                if re.match(pattern, s[i][0]):  # pattern 뒤의 조사 찾기
                    is_qus = True
                    if s[i][0] == '몇':
                        is_m = True
                        index = sentence.find('몇')
                        m_word = ""
                        for k in range(index + 2, len(sentence)):
                            if sentence[k] in origin:
                                m_word += sentence[k]
                            else:
                                break

                        if m_word in origin and len(m_word) > 0: # 몇 시, 몇 일 처리
                            pattern = pattern.replace('|', " "+m_word + '|')

                    if s[i + 1][1] in josa:
                        josa_word = s[i + 1][0]
                        is_josa = True
                        try:
                            right_josa = Josa.get_full_string(answers[j], josa_word)  # 올바른 조사
                            jos_pattern = pattern.replace('|', josa_word + '|')
                        except:
                            is_josa = False
                        break
                    break

            pos_sentence = mecab.pos(sentence)
            match = re.match(pattern, pos_sentence[-1][0])

            if match:   # 끝문자에 pattern이 있을 경우 pattern뺴고 처리
                list_pos = list(pos_sentence[-1])
                list_pos[0] = re.sub(pattern, "", list_pos[0])
                pos_sentence[-1] = tuple(list_pos)

            if is_m and len(m_word) > 0:
                m_compile = re.compile(m_word)  # 몇 뒤에 단어가 끝인 경우
                m_match = m_compile.search(sentence)
                if m_match.span():
                    span = m_match.span()
                    last_index = sentence.find(pos_sentence[-1][0])
                    if span[0] <= last_index < span[1]:
                        list_pos = list(pos_sentence[-1])
                        list_pos[0] = sentence[span[1]:]
                        pos_sentence[-1] = tuple(list_pos)


            # 끝 문자 바꾸기
            if 'VCP' in pos_sentence[-1][1]:
                if pos_sentence[-2][0] in pattern:
                    if krx_api.has_conda(answers[j]):   # 종성있으면 이다
                        sentence = re.sub(pos_sentence[-1][0] + "$", '이다.', sentence)
                    else:
                        sentence = re.sub(pos_sentence[-1][0] + "$", '다.', sentence)
                else:
                    if krx_api.has_conda(pos_sentence[-2][0]):
                        sentence = re.sub(pos_sentence[-1][0] + "$", '이다.', sentence)
                    else:
                        sentence = re.sub(pos_sentence[-1][0] + "$", '다.', sentence)

            elif 'VV' in pos_sentence[-1][1]:
                if pos_sentence[-1][0] == "돼":
                    if pos_sentence[-2][1] == 'MAG':
                        sentence = re.sub(pos_sentence[-1][0] + "$", '이다.', sentence)
                    else:
                        sentence = re.sub(pos_sentence[-1][0] + "$", '된다.', sentence)
                else:
                    sentence = re.sub(pos_sentence[-1][0] + "$", '한다.', sentence)

            elif 'XSV' in pos_sentence[-1][1]:
                sentence = re.sub(pos_sentence[-1][0] + "$", '되다.', sentence)

            elif 'VX' in pos_sentence[-1][1]:
                sentence = re.sub(pos_sentence[-1][0] + "$", '하다.', sentence)
            else:
                sentence = re.sub(pos_sentence[-1][0] + "$", '다.', sentence)

            if is_qus:
                # print('is_qus!', sentence, pattern, jos_pattern)
                if re.search(jos_pattern, sentence) and is_josa:
                    # print('Im here!!!')
                    sentence = re.sub(jos_pattern, right_josa, sentence, 1)
                elif re.search(pattern, sentence):
                    sentence = re.sub(pattern, answers[j], sentence, 1)
                else:
                    if krx_api.has_conda(answers[0]):
                        sentence = sentence + ' ' + answers[0] + '이다.'
                    else:
                        sentence = sentence + ' ' + answers[0] + '다.'
            else:
                if krx_api.has_conda(answers[0]):
                    sentence = sentence + ' ' + answers[0] + '이다.'
                else:
                    sentence = sentence + ' ' + answers[0] + '다.'

            sentence_list.append(sentence)
        print('sentence list:', sentence_list)

        # answer 바꾸기
        if is_exist and is_qus:
            replace_answer = masking.main(sentence_list, choice_answer, answer_sibling)
            # replace_answer = self.test_mask(sentence, choice_answer, answer_sibling)
            for j in range(len(sentence_list)):
                    answers[j] = answers[j].replace(choice_answer[j][0], replace_answer[j], 1)
                    sentence_list[j] = sentence_list[j].replace(origin, answers[j], 1)
                    print("답:", origin, '->', answers[j])

        return sentence_list

    def transfer_sentence_to_plain(self, question, answer, change_answer):
        pattern = "누가|누구|누굴|누군지|언제|어디|어딜|며칠|며칟날|무슨|무엇|무얼|뭐|뭘|뭔|얼마나|얼마|얼만큼|어떤|어느|왜|어쨰서|어떻게|어떻|어떠|어땠|어때|어떨|몇|어찌하여"
        josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', ' JKQ', 'JX', ' JC']  # 정답
        sign = ['SF']  # 질문에서 끝 문자 ?,! 등등... 제외

        is_qus, is_m, is_josa = False, False, False  # pattern에 매칭되는지, 몇에 매칭 되는지, 조사에 매칭 되는지

        temp_pattern, josa_pattern = pattern[:], ""  # pattern 복사
        # delete sign
        for w, p in self.mecab.pos(question):
            if p in sign:
                question = question.replace(w, "")

        # pattern 수정이 필요한 경우 몇 뒤에 단어와 조사 수정
        pos_question = self.mecab.pos(question)
        for i in range(len(pos_question) - 1):
            if re.match(pattern, pos_question[i][0]):  # pattern에 매칭될 경우
                is_qus = True
                if pos_question[i][0] == '몇':
                    is_m = True
                    m_index = question.find('몇')
                    m_word = ""  # 몇 뒤의 단어 저장할 변수
                    if question[m_index + 1] == " ":  # 몇 뒤에 띄어져있으면 m_word시작 부분에 띄어쓰기 추가
                        m_word = " "
                    for j in range(m_index + 1, len(question)):  # 기계데이터는 index+1
                        if question[j] in answer:
                            m_word += question[j]
                        else:
                            break

                    if m_word.strip() in answer and len(m_word) > 0:  # 몇 뒤에 단어 처리
                        pattern = pattern.replace('|', m_word + '|')

                if pos_question[i + 1][1] in josa:  # pattern뒤에 조사가 오는경우
                    josa_word = pos_question[i + 1][0]
                    is_josa = True
                    try:
                        right_josa = Josa.get_full_string(change_answer, josa_word)  # 변형될 정답의 올바른 조사 찾기
                        josa_pattern = pattern.replace('|', josa_word + '|')
                    except:
                        is_josa = False

        # 끝 문자에 pattern이 포함될 경우 pattern빼고 처리
        if re.match(temp_pattern, pos_question[-1][0]):
            list_pos = list(pos_question[-1])
            list_pos[0] = re.sub(temp_pattern, "", list_pos[0])
            pos_question[-1] = tuple(list_pos)

        # 몇 뒤에 단어가 끝인 경우 처리
        if is_m and len(m_word) > 0:
            m_compile = re.compile(m_word)
            if m_compile.search(question):
                m_match = m_compile.search(question)
                span = m_match.span()
                last_index = question.find(pos_question[-1][-0])
                if span[0] <= last_index < span[1]:
                    list_pos = list(pos_question[-1])
                    list_pos[0] = question[span[1]:]
                    pos_question[-1] = tuple(list_pos)

        # 끝 문자 바꿔서 평서문 변환
        if 'VCP' in pos_question[-1][1]:
            if pos_question[-2][0] in pattern:
                if krx_api.has_conda(change_answer):
                    sentence = re.sub(pos_question[-1][0] + "$", "이다.", question)
                else:
                    sentence = re.sub(pos_question[-1][0] + "$", '다.', question)
            else:
                if krx_api.has_conda(pos_question[-2][0]):
                    sentence = re.sub(pos_question[-1][0] + "$", "이다.", question)
                else:
                    sentence = re.sub(pos_question[-1][0] + "$", '다.', question)

        elif 'VV' in pos_question[-1][1]:
            if pos_question[-1][0] == "돼":
                if pos_question[-2][1] == 'MAG':
                    sentence = re.sub(pos_question[-1][0] + "$", '이다.', question)
                else:
                    sentence = re.sub(pos_question[-1][0] + "$", '된다.', question)
            else:
                sentence = re.sub(pos_question[-1][0] + "$", '한다.', question)

        elif 'XSV' in pos_question[-1][1]:
            sentence = re.sub(pos_question[-1][0] + "$", '되다.', question)

        elif 'VX' in pos_question[-1][1]:
            sentence = re.sub(pos_question[-1][0] + "$", '하다.', question)
        else:
            sentence = re.sub(pos_question[-1][0] + "$", '다.', question)

        # 답 질문에 대치
        if is_qus:
            if re.search(josa_pattern, sentence) and is_josa:  # 조사 패턴 캐칭
                sentence = re.sub(josa_pattern, right_josa, sentence, 1)
            elif re.search(pattern, sentence):  # 패턴 매칭
                sentence = re.sub(pattern, change_answer, sentence, 1)
            else:
                if krx_api.has_conda(change_answer):
                    sentence = sentence + " " + change_answer + "이다."
                else:
                    sentence = sentence + " " + change_answer + "다."

        else:
            if krx_api.has_conda(change_answer):
                sentence = sentence + " " + change_answer + "이다."
            else:
                sentence = sentence + " " + change_answer + "다."

        return sentence

    def load_file(self, f):
        test = False
        f = '기계독해'

        file_name = 'ko_nia_normal_squad_all.json'
        self.labeled_train_files = [file_name]

        context, questions, answer, answer_starts = [], [], [], []

        for file_path in self.labeled_train_files:
            with open(file_path, 'r', encoding='UTF-8') as file:    # json 파일 로드
                json_string = json.load(file)
                datas = json_string['data']
                count = 0
                # if test:
                #     break
                for data in datas: # data context, question, answer 읽기
                    # count += 1
                    # if count > 9:
                    #     test = True
                    #     break

                    if f == '도서':
                        for para in data['paragraphs']:
                            for qas in para['qas']:
                                questions.append(qas['question'])
                                context.append([para['context']])
                                answer.append(qas['answers'][0]['text'])
                                answer_starts.append(qas['answers'][0]['answer_start'])
                    elif f.find('기계독해') != -1:
                        for para in data['paragraphs']:
                            qas = para['qas']
                            for q in qas:
                                questions.append(q['question'])
                                context.append([para['context']])
                                answer.append(q['answers'][0]['text'])
                                answer_starts.append(q['answers'][0]['answer_start'])
                    else:
                        qas = data['paragraphs'][0]['qas']
                        for q in qas:
                            questions.append(q['question'])
                            context.append([data['paragraphs'][0]['context']])
                            answer.append(q['answers']['text'])
                            answer_starts.append(q['answers']['answer_start'])

        return context, questions, answer, answer_starts

    def extract_siblings_for_data(self, data):
        index, context, question, answer, answer_start = data
        nouns, sibling = self.find_noun(question)  # 명사와 형제어 찾기
        # if len(nouns) < 1:      # 명사가 없다면
        #    print("명사가 없습니다.")
        #    continue

        """
        answer sibling extract
        """
        choice_answer, answer_sibling = [], []

        # answer 형제어 바꾸기
        origin = answer
        noun_answer = self.mecab.nouns(answer)
        # Korlex 없는 단어 제외
        words = noun_answer[:]
        sibling_list = []
        for i, word in enumerate(noun_answer):
            if word not in self.krx:
                sibling_list.append(self.extract_sibling(word))
                self.krx[word] = sibling_list[i]
                if not sibling_list[i]:
                    words.remove(word)
            else:
                if len(self.krx[word]) > 30:
                    sibling_list.append(random.sample(self.krx[word], 30))
                else:
                    sibling_list.append(self.krx[word])

                if not sibling_list[i]:
                    words.remove(word)

        if len(words) > 0:
            choice_answer = random.sample(words, 1)[0]
            n_index = noun_answer.index(choice_answer)
            answer_sibling.append(sibling_list[n_index])
        else:
            is_exist = False

        if len(answer_sibling) > 0:
            answer_sibling = answer_sibling[0]

        print(index, nouns, sibling, choice_answer, answer_sibling)
        return (nouns, sibling, choice_answer, answer_sibling, context, question, answer, answer_start)


def check(question, answer):
    print(question, answer)


if __name__ == '__main__':
    file = "기계독해_valid"  # input()

    aug = Augmentation('./corpus/*'+file+'*/**/*.json')
    contexts, questions, answers, answer_starts = aug.load_file(file)

    data = list(zip(range(len(questions)), contexts, questions, answers, answer_starts))

    pool = multiprocessing.Pool(processes=128)
    results = pool.map(
        aug.extract_siblings_for_data,
        data[len(data) - 20000:len(data)]
    )
    pool.close()
    pool.join()

    result_file = open(file + '_result_file.txt', 'w', encoding='utf-8')

    for r in range(len(results)):
        nouns, siblings, choice_answer, answer_sibling, context, question, answer, answer_start = results[r]

        for noun, sibling in zip(nouns, siblings):
            line = noun
            #print(line)
            for word in sibling:
                line += '\t' + word
            result_file.write(line + '\n')

        if len(choice_answer) == 0:
            choice_answer = ''
        line = choice_answer

        for word in answer_sibling:
            line += '\t' + word
        result_file.write(line + '\n')
        #print(answer, answer_start)
        result_file.write(question.replace('\n', ' ') + '\n')
        result_file.write(context[0].replace('\n', ' ') + '\n')
        result_file.write(answer.replace('\n', ' ') + '\n')
        result_file.write(str(answer_start).replace('\n', ' '))
        result_file.write('\t\n\n\t')
    print(len(data))
    print(len(results))
