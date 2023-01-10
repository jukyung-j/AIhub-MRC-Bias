import re
import time
import pickle
import pandas as pd
import numpy as np
import copy
import os
import jsonlines
import json

# from sentence_transformers import SentenceTransformer, util
from konlpy.tag import Mecab
from pyjosa.josa import Josa

## Korlex API Definition
from krx_def import *


mecab = Mecab()
class KorLexAPI:
    ### PRIVATE ###
    def __init__(self, ssInfo_path:str, seIdx_path:str, reIdx_path:str):
        print("[KorLexAPI][INIT] Plz Wait...")

        self.is_set_ssInfo_path = False
        self.is_set_seIdx_path = False
        self.is_set_reIdx_path = False

        # check seIdx_path
        if 0 >= len(seIdx_path):
            print("[KorLexAPI][INIT] ERR - Plz check seIdx_path:", seIdx_path)
            return
        if not os.path.exists(seIdx_path):
            print("[KorLexAPI][INIT] ERR -", seIdx_path, "is Not Existed !")
            return

        self.seIdx_path = seIdx_path
        self.is_set_seIdx_path = True

        # check reIdx_path
        if 0 >= len(reIdx_path):
            print("[KorLexAPI][INIT] ERR - Plz check reIdx_path:", reIdx_path)
            return
        if not os.path.exists(reIdx_path):
            print("[KorLexAPI][INIT] ERR -", reIdx_path, "is Not Existed !")
            return

        self.reIdx_path = reIdx_path
        self.is_set_reIdx_path = True

        # Check ssinfo_path
        if 0 >= len(ssInfo_path):
            print("[KorLexAPI][INIT] ERR - Plz check ssInfo_path:", ssInfo_path)
            return

        if not os.path.exists(ssInfo_path):
            print("[KorLexAPI][INIT] ERR -", ssInfo_path, "is Not Existed !")
            return

        self.is_set_ssInfo_path = True
        self.ssInfo_path = ssInfo_path
        print("[KorLexAPI][INIT] - Complete set to path,", self.ssInfo_path,
              "you can use load method.")

    def _make_sibling_list(self, soff:int, pos:str):
        ret_sibling_list = []

        target_re_idx_list = np.where((self.reIdx_df["elem"].values == soff) &
                                        (self.reIdx_df["relation"].values == "child") &
                                        (self.reIdx_df["trg_pos"].values == pos))
        for t_elem_re_idx in target_re_idx_list:
            for _, reIdx_item in self.reIdx_df.loc[t_elem_re_idx].iterrows():
                trg_elem = reIdx_item["trg_elem"]
                trg_elem_seIdx_list = np.where((self.seIdx_df["soff"].values == trg_elem) &
                                               (self.seIdx_df["pos"].values == pos))

                for t_elem_se_idx in trg_elem_seIdx_list:
                    se_ss_node = SS_Node([], soff=trg_elem, pos=pos)
                    for _, seIdx_item in self.seIdx_df.loc[t_elem_se_idx].iterrows():
                        se_synset = Synset(text=seIdx_item["word"],
                                           sense_id=seIdx_item["senseid"])
                        se_ss_node.synset_list.append(copy.deepcopy(se_synset))
                    ret_sibling_list.append(copy.deepcopy(se_ss_node))

        return ret_sibling_list

    def _make_result_json(self, target_obj:object, ontology:str):
        ret_korlex_result_list = []

        # check, is target parent dobule?
        target_parent_list = []
        check_parent_list = np.where(self.reIdx_df["trg_elem"].values == target_obj["soff"])
        for pt_idx in check_parent_list:
            for _, pt_item in self.reIdx_df.loc[pt_idx].iterrows():
                pt_relation = pt_item["relation"]
                if "child" == pt_relation:
                    pt_elem = pt_item["elem"]
                    pt_pos = pt_item["pos"]
                    target_parent_list.append((pt_elem, pt_pos))

        if 0 >= len(target_parent_list): # Except (e.g. eat(convert to korean))
            result_data = KorLexResult(Target(ontology=ontology,
                                                  word=target_obj["word"],
                                                  pos=target_obj["pos"],
                                                  sense_id=target_obj["senseid"],
                                                  soff=target_obj["soff"]), [], [])

            ss_node = SS_Node(synset_list=[], soff=target_obj["soff"], pos=target_obj["pos"])
            seIdx_matching_list = np.where(self.seIdx_df["soff"].values == ss_node.soff)
            for mat_idx in seIdx_matching_list:
                for _, seIdx_item in self.seIdx_df.loc[mat_idx].iterrows():
                    seIdx_word = seIdx_item["word"]
                    seIdx_pos = seIdx_item["pos"]
                    seIdx_senseId = seIdx_item["senseid"]

                    if seIdx_pos == target_obj["pos"]:
                        synset_data = Synset(text=seIdx_word, sense_id=seIdx_senseId)
                        ss_node.synset_list.append(copy.deepcopy(synset_data))
            result_data.results.append(ss_node)

            sibling_list = self._make_sibling_list(soff=target_obj["soff"], pos=target_obj["pos"])
            result_data.siblings = copy.deepcopy(sibling_list)

            ret_korlex_result_list.append(result_data)

        # Existed Parent
        for target_parent in target_parent_list:
            # set target info
            result_data = KorLexResult(Target(ontology=ontology,
                                                  word=target_obj["word"],
                                                  pos=target_obj["pos"],
                                                  sense_id=target_obj["senseid"],
                                                  soff=target_obj["soff"]), [], [])

            ## Search processing
            curr_target = (target_parent[0], target_parent[-1])

            # current target synset
            curr_ss_node = SS_Node(synset_list=[], soff=target_obj["soff"], pos=target_obj["pos"])
            curr_se_matcing_list = np.where((self.seIdx_df["soff"].values == target_obj["soff"]) &
                                            (self.seIdx_df["pos"].values == target_obj["pos"]))

            for curr_se_idx in curr_se_matcing_list:
                for _, curr_se_item in self.seIdx_df.loc[curr_se_idx].iterrows():
                    curr_seIdx_word = curr_se_item["word"]
                    curr_seIdx_senseId = curr_se_item["senseid"]
                    curr_synset_data = Synset(text=curr_seIdx_word, sense_id=curr_seIdx_senseId)
                    curr_ss_node.synset_list.append(copy.deepcopy(curr_synset_data))
            result_data.results.append(curr_ss_node)

            # search sibling for target
            sibling_list = self._make_sibling_list(soff=curr_target[0], pos=curr_target[-1])
            result_data.siblings = copy.deepcopy(sibling_list)

            # search loop
            while True:
                prev_target = copy.deepcopy(curr_target)

                # Search synset
                ss_node = SS_Node(synset_list=[], soff=curr_target[0], pos=curr_target[-1])
                seIdx_matching_list = np.where(self.seIdx_df["soff"].values == curr_target[0])
                for mat_idx in seIdx_matching_list:
                    for _, seIdx_item in self.seIdx_df.loc[mat_idx].iterrows():
                        seIdx_word = seIdx_item["word"]
                        seIdx_pos = seIdx_item["pos"]
                        seIdx_senseId = seIdx_item["senseid"]

                        if seIdx_pos == curr_target[-1]:
                            synset_data = Synset(text=seIdx_word, sense_id=seIdx_senseId)
                            ss_node.synset_list.append(copy.deepcopy(synset_data))

                if 0 >= len(ss_node.synset_list):
                    break
                else:
                    result_data.results.append(copy.deepcopy(ss_node))

                # Search parent
                reIdx_matching_list = np.where(self.reIdx_df["trg_elem"].values == curr_target[0])
                for mat_idx in reIdx_matching_list:
                    for _, reIdx_item in self.reIdx_df.loc[mat_idx].iterrows():
                        reIdx_rel = reIdx_item["relation"]
                        reIdx_pos = reIdx_item["pos"]

                        if ("child" == reIdx_rel) and (reIdx_pos == curr_target[-1]):
                            reIdx_elem = reIdx_item["elem"]
                            curr_target = (reIdx_elem, reIdx_pos)
                            break

                if(prev_target[0] == curr_target[0]): break
            ret_korlex_result_list.append(copy.deepcopy(result_data))

        return ret_korlex_result_list

    ### PUBLIC ###
    def load_synset_data(self):
        print("[KorLexAPI][load_synset_data] Load JSON Data, Wait...")
        is_set_pkl_files = True
        if not self.is_set_ssInfo_path:
            print("[KorLexAPI][load_synset_data] ERR - Plz set json path")
            is_set_pkl_files = False

        if not self.is_set_seIdx_path:
            print("[KorLexAPI][load_synset_data] ERR - Plz set seIdx path")
            is_set_pkl_files = False

        if not self.is_set_reIdx_path:
            print("[KorLexAPI][load_synset_data] ERR - Plz set reIdx path")
            is_set_pkl_files = False

        if not is_set_pkl_files: return

        # Load seIdx.pkl
        print("[KorLexAPI][load_synset_data] Loading seIdx.pkl...")
        self.seIdx_df = None
        with open(self.seIdx_path, mode="rb") as seIdx_file:
            self.seIdx_df = pickle.load(seIdx_file)
            print("[KorLexAPI][load_synset_data] Loaded seIdx.pkl !")

        # Load reIdx.pkl
        print("[KorLexAPI][load_synset_data] Loading reIdx.pkl...")
        self.reIdx_df = None
        with open(self.reIdx_path, mode="rb") as reIdx_file:
            self.reIdx_df = pickle.load(reIdx_file)
            print("[KorLexAPI][load_synset_data] Loaded reIdx.pkl !")

        # Load ssInfo
        print("[KorLexAPI][load_synset_data] Loading ssInfo.pkl...")
        self.ssInfo_df = None
        with open(self.ssInfo_path, mode="rb") as ssInfo_file:
            self.ssInfo_df = pickle.load(ssInfo_file)
            print("[KorLexAPI][load_synset_data] Loaded ssInfo.pkl !")

    def search_word(self, word:str, ontology=str):
        ret_json_list = []

        if 0 >= len(word):
            print("[KorLexAPI][search_word] ERR - Check input:", word)
            return ret_json_list

        if word not in self.seIdx_df["word"].values:
            print("[KorLexAPI][search_word] ERR - Not Existed SE Index Table:", word)
            return ret_json_list

        # Search sibling nodes
        sibling_idx_list = np.where(self.seIdx_df["word"].values == word)
        sibling_obj_list = []
        for sIdx in sibling_idx_list[0]:
            sibling_obj_list.append(copy.deepcopy(self.seIdx_df.loc[sIdx]))

        # Make Result Json
        for target_obj in sibling_obj_list:
            target_krx_json = self._make_result_json(target_obj=target_obj, ontology=ontology)
            ret_json_list.append(copy.deepcopy(target_krx_json))

        return ret_json_list

    def search_synset(self, synset:str, ontology:str):
        ret_json_list = []

        if 0 >= len(synset):
            print("[KorLexAPI][search_synset] ERR - Check input:", synset)
            return ret_json_list

        synset = int(synset)
        if synset not in self.seIdx_df["soff"].values:
            print("[KorLexAPI][search_synset] ERR - Not Existed SE Index Table:", synset)
            return ret_json_list

        # Search sibling nodes
        sibling_idx_list = np.where(self.seIdx_df["soff"].values == synset)
        sibling_obj_list = []
        for sIdx in sibling_idx_list[0]:
            sibling_obj_list.append(copy.deepcopy(self.seIdx_df.loc[sIdx]))

        # Make Result Json
        for target_obj in sibling_obj_list:
            target_krx_json = self._make_result_json(target_obj=target_obj, ontology=ontology)
            ret_json_list.append(copy.deepcopy(target_krx_json))

        return ret_json_list

def has_conda(word: str):    #아스키(ASCII) 코드 공식에 따라 입력된 단어의 마지막 글자 받침 유무를 판단해서 뒤에 붙는 조사를 리턴하는 함수
    last = word[-1]     #입력된 word의 마지막 글자를 선택해서
    criteria = (ord(last) - 44032) % 28     #아스키(ASCII) 코드 공식에 따라 계산 (계산법은 다음 포스팅을 참고하였습니다 : http://gpgstudy.com/forum/viewtopic.php?p=45059#p45059)
    if criteria == 0: #종성없음
        return False
    else: #종성있음
        return True

def change_sibling(word, wd:str, stn: str):
    ## Original Setnence
    sentence_list = ""

    if stn.find(wd) + len(wd) < len(stn):
        if has_conda(word): ## 종성있으면 은/이/을/과
            if stn[stn.find(wd) + len(wd)] == '는':
                modified_sentence = stn.replace(f'{wd}는', f'{word}은', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '를':
                modified_sentence = stn.replace(f'{wd}를', f'{word}을', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '가':
                modified_sentence = stn.replace(f'{wd}가', f'{word}이', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '와':
                modified_sentence = stn.replace(f'{wd}와', f'{word}과', 1)
                sentence_list = modified_sentence
            else:
                modified_sentence = stn.replace(wd, word, 1)
                sentence_list = modified_sentence
        else:  ## 종성없으면 는/가/를/와
            if stn[stn.find(wd) + len(wd)] == '은':
                modified_sentence = stn.replace(f'{wd}은', f'{word}는', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '을':
                modified_sentence = stn.replace(f'{wd}을', f'{word}를', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '이':
                modified_sentence = stn.replace(f'{wd}이', f'{word}가', 1)
                sentence_list = modified_sentence
            elif stn[stn.find(wd) + len(wd)] == '과':
                modified_sentence = stn.replace(f'{wd}과', f'{word}와', 1)
                sentence_list = modified_sentence
            else:
                modified_sentence = stn.replace(wd, word, 1)
                sentence_list = modified_sentence
    else:
        modified_sentence = stn.replace(wd, word, 1)
        sentence_list = modified_sentence

    return sentence_list

def extract_en(word_list):

    new_word_list = []
    for word in word_list:
        w = re.sub('[a-zA-Z]', '', word)
        if w==word:
            new_word_list.append(word)

    return new_word_list

def  transfer_sentence(sentence, answer):

    pattern =  "누가|누구|누굴|누군지|언제|어디|어딜|몇|며칠|며칟날|무슨|무엇|무얼|뭐|뭘|뭔|얼마|얼만큼|어떤|어느|왜|어쨰서|어찌하여|어떻|어떠|어땠|어때|어떨"
    josa = ['JKS', 'JKC', 'JKG', 'JKO','JKB', 'JKV',' JKQ', 'JX',' JC']

    s = mecab.pos(sentence)
    is_josa = False

    for i in range(len(s) - 1): # pattern 뒤의 조사 찾기
        if s[i][0] in pattern:
            if s[i+1][1] in josa:
                josa_word = s[i+1][0]
                is_josa = True
                try:
                    right_josa = Josa.get_full_string(answer, josa_word)  # 올바른 조사
                    pattern = pattern.replace('|', josa_word + '|')
                except:
                    is_josa = False

                break

    sentence = re.sub(pattern, right_josa, sentence) if is_josa else re.sub(pattern, answer, sentence)
    pos_sentence = mecab.pos(sentence)

    if 'VCP' in pos_sentence[-1][1]:
        sentence = re.sub(pos_sentence[-1][0]+"$", '이다', sentence)
    elif 'VV' in pos_sentence[-1][1]:
        sentence = re.sub(pos_sentence[-1][0]+"$", '한다', sentence)
    else:
        sentence = re.sub(pos_sentence[-1][0]+"$", '다', sentence)


    return sentence

### TEST ###
if "__main__" == __name__:
    ## Load Files
    ssInfo_path = "./dic/korlex_ssInfo.pkl"
    seIdx_path = "./dic/korlex_seIdx.pkl"
    reIdx_path = "./dic/korlex_reIdx.pkl"
    krx_json_api = KorLexAPI(ssInfo_path=ssInfo_path,
                             seIdx_path=seIdx_path,
                             reIdx_path=reIdx_path)
    krx_json_api.load_synset_data()

    # test_search_synset = krx_json_api.search_word(word='오늘', ontology=ONTOLOGY.KORLEX.value)
    # for t_s in test_search_synset:
    #     print(t_s)

    ## Enter the number of words to be replaced
    n = int(input('변경할 명사 몇 개?: '))
    sentence = '어디서 보건소 임기제공무원 의무5급 신규채용 및 채용의뢰를 하지'
    answer = '서울특별시 각종 위원회의 설치·운영에 관한 조례'
    all_sentences = []
    stn_list = []


    # 평서문 변환
    sentence = transfer_sentence(sentence, answer)
    print(sentence)
    # Iterate n-turns, Enter words to be replaced
    for i in range(n):
        word = input('바꿀 단어?: ')
        test_search_synset = krx_json_api.search_word(word=word, ontology=ONTOLOGY.KORLEX.value)

        print(test_search_synset)
        ## Extract Siblings
        word_list = []
        for t_s in test_search_synset:
            sub_siblings = str(t_s[0]).split('siblings=')[1:]

            for sub in sub_siblings:
                sib_list = sub.split('KorLexResult(')[0].split('text=')

                for sib in sib_list:
                    m = re.search('\'(.+?)\', sense', sib)
                    if m:
                        found = m.group(1)
                        word_list.append(found)

        word_list = extract_en(word_list)

        ## Run change_sibling
        if len(all_sentences)==0:
            all_sentences = change_sibling(word_list, word, sentence)
        else:
            for all in all_sentences:
                tmp_list = change_sibling(word_list, word, all)
                for tmp in tmp_list:
                    stn_list.append(tmp)
            for stn in stn_list:
                all_sentences.append(stn)
        print()



    ## Make Model
    # model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    # embeddings = model.encode(all_sentences, convert_to_tensor=True)
    # original_embeddings = model.encode(sentence, convert_to_tensor=True)
    # cosine_scores = util.pytorch_cos_sim(embeddings, original_embeddings)
    #
    # cosine_scores_list = cosine_scores.tolist()
    # text_and_cosine = {}
    #
    # for text, cosine in zip(all_sentences, cosine_scores_list):
    #     text_and_cosine[text] = cosine
    # text_and_cosine_ranked = sorted(text_and_cosine.items(), key=lambda x: x[1])

    ## Save text and cosine_similarity
    # with open("C:\\Users\\helen\\Desktop\\연구과제 관련\\AIHUB\\text_and_cosine.jsonl", encoding="utf-8", mode = 'w') as file:
    #     for data in text_and_cosine:
    #         print(data)
    #         file.write(json.dumps(data, ensure_ascii=False) + "\n")
    # with open("C:\\Users\\helen\\Desktop\\연구과제 관련\\AIHUB\\text_and_cosine_rankded.jsonl", encoding="utf-8", mode = 'w') as file:
    #     for data in text_and_cosine_ranked:
    #         file.write(json.dumps(data, ensure_ascii=False) + "\n")
    #
    # print('Finish!!')

    ## Save Tensor

    exit()
