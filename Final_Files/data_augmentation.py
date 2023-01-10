import glob
import os
import re
import json
import random
from konlpy.tag import Mecab
import krx_api
from pyjosa.josa import Josa
import masking


class Augmentation:
    def __init__(self, file_format):
        self.mecab = Mecab()
        self.labeled_train_files, self.labeled_valid_files, self.origin_train_files, self.origin_valid_files = [], [], [], []
        for name in glob.glob(file_format, recursive=True):  # 데이터 파일 가져오기
            if '라벨링데이터' in name:
                if 'Training' in name:
                    self.labeled_train_files.append(name)
                else:
                    self.labeled_valid_files.append(name)
            elif '원천데이터' in name:
                if 'Training' in name:
                    self.origin_train_files.append(name)
                else:
                    self.origin_valid_files.append(name)
            else:
                if 'Training' in name:
                    self.labeled_train_files.append(name)
                else:
                    self.labeled_valid_files.append(name)
        ## Load Korlex Files
        ssInfo_path = "./dic/korlex_ssInfo.pkl"
        seIdx_path = "./dic/korlex_seIdx.pkl"
        reIdx_path = "./dic/korlex_reIdx.pkl"
        self.krx_json_api = krx_api.KorLexAPI(ssInfo_path=ssInfo_path,
                                 seIdx_path=seIdx_path,
                                 reIdx_path=reIdx_path)
        self.krx_json_api.load_synset_data()
        self.krx = {}

    def extract_sibling(self, word):
        word_list, replace_list = [], []

        if word not in self.krx:
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

        if len(word_list) > 10:
            word_list = random.sample(word_list, 10)
        return word_list

    def find_noun(self, sentence):
        is_noun = False
        word, nouns = "", []
        pattern = ["누가","누구","누굴","누군지","언제","어디","어딜","며칠","며칟날","무슨","무엇","무얼","뭐","뭘","뭔","얼마나","얼마","얼만큼","어떤","어느","왜","어쨰서","어떻게","어떻","어떠","어땠","어때","어떨","몇","어찌하여"]

        is_m = False

        pos_sentence = self.mecab.pos(sentence)
        sentence = sentence.split()
        index = 0
        pos_word = ""
        is_continue = False

        for w, p in pos_sentence:
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

            if not sibling_list[i]:
                words.remove(word)
        if len(words) > 0:
            choice_noun = random.sample(words, 1)
            index = nouns.index(choice_noun[0])
            sibling = sibling_list[index]
        else:
            choice_noun = []
            sibling = []

        return choice_noun, sibling

    def test_mask(self, question, origin_word, sibling_list):
        word = [0 for _ in range(len(sibling_list))]

        for i in range(len(sibling_list)):
            word[i] = sibling_list[i][0]

        return word

    def transfer_sentence(self, sentences, answers):  # 평서문 바꾸기
        mecab = Mecab()
        pattern = "누가|누구|누굴|누군지|언제|어디|어딜|며칠|며칟날|무슨|무엇|무얼|뭐|뭘|뭔|얼마나|얼마|얼만큼|어떤|어느|왜|어쨰서|어떻게|어떻|어떠|어땠|어때|어떨|몇|어찌하여"
        josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', ' JKQ', 'JX', ' JC']
        sign = ['SF']  # 질문에서 끝 문자 ?,! 등등... 제외
        temp = pattern[:]
        sentence_list = []
        choice_answer, answer_sibling = [], []
        is_exist = True
        jos_pattern = ''

        for j in range(len(answers)):
            # answer 형제어 바꾸기
            origin = answers[j]
            noun_answer = self.mecab.nouns(answers[j])
            print('zz: ', noun_answer)
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
        print("wo; ", answer_sibling)
        is_qus = False
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

                        if m_word in origin and len(m_word) > 0:  # 몇 시, 몇 일 처리
                            pattern = pattern.replace('|', " " + m_word + '|')

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

            if match:  # 끝문자에 pattern이 있을 경우 pattern뺴고 처리
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
                    if krx_api.has_conda(answers[j]):  # 종성있으면 이다
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

    def load_file(self, f):
        test = False
        for file_path in self.labeled_train_files:
            context, questions, answer, qas = [], [], [], []
            file_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='UTF-8') as file:
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
                            qt, aw = [], []
                            context.append([para['context']])
                            for qas in para['qas']:
                                qt.append(qas['question'])
                                aw.append(qas['answers'][0]['text'])
                            questions.append(qt)
                            answer.append(aw)

                    else:
                        context.append([data['paragraphs'][0]['context']])
                        qas = data['paragraphs'][0]['qas']
                        for q in qas:
                            qt.append(q['question'])
                            aw.append(q['answers']['text'])
                        questions.append(qt)
                        answer.append(aw)

            # questions = [['일본의 식민 지배를 정당화하는 등 왜곡을 일삼자, 정읍시의회가 강하게 비판한 하버드대 교수는 누구야', '정읍시의회는 일본의 식민지배를 옹호하는 발언을 한 누구에게 강한 유감을 표했어']]
            # answer = [['존 램지어', '존 램지어']]

            for index, question in enumerate(questions):
                # 명사 찾아서 랜덤 하게 선택
                siblings_list, choice_nouns = [], []
                replace_sentence = []
                replace_word = []
                qt = question[:]

                for qus in qt:
                    nouns, sibling = self.find_noun(qus)    # 명사 찾기
                    if len(nouns) < 1:      # 명사가 없다면
                        print("명사가 없습니다.")
                        question.remove(qus)
                        continue
                    choice_nouns.append(nouns)
                    siblings_list.append(sibling)
                print("원시문장:", question)
                print("선택된 단어 : ", choice_nouns)
                if not choice_nouns:
                    print("명사 없음 pass")
                    continue

                # for word in choice_nouns:
                #     siblings_list.append(self.extract_sibling(word))

                replace_word = masking.main(question, choice_nouns, siblings_list)   # 순위 높은 형제어 받아오기
                # replace_word = self.test_mask(question, choice_nouns, siblings_list)   # 순위 높은 형제어 받아오기

                print("대체될 단어: ",replace_word)

                for i in range(len(question)):
                    replace_sentence.append(krx_api.change_sibling(replace_word[i], choice_nouns[i][0], question[i]))   # 형제어 치환

                replace_sentence = self.transfer_sentence(replace_sentence, answer[index])  # 평서문 바꾸기
                print("대치된 문장:", replace_sentence)

                # context에 랜덤하게 대치된 문장 삽입
                split_context = context[index][0].replace('다. ', '다.\n').split('\n')
                context_length = len(split_context)
                for s in replace_sentence:
                    random_index = random.randrange(context_length)
                    split_context.insert(random_index, s)

                edited_context = ""
                for s in split_context:
                    edited_context += s+' '

                context[index][0] = edited_context
                # print(context[index])
                print()

            make_file = open('./corpus/'+'edited_'+f+'/edited_'+file_name, 'w', encoding='utf-8')
            file = open(file_path, 'r', encoding='utf-8')
            json_string = json.load(file)

            for i, data in enumerate(json_string['data']):
                data['paragraphs'][0]['context'] = context[i][0]

            file.close()
            json.dump(json_string, make_file, indent='\t', ensure_ascii=False)

            make_file.close()

if __name__ == '__main__':
    while True:
        print("행정, 뉴스, 도서중 입력 : ")
        file = input()
        if file == '행정' or file == '뉴스' or file == '도서':
            break

    aug = Augmentation('./corpus/*'+file+'*/**/*.json')
    aug.load_file(file)
