import os

from squad_utils import *
from random import randint

import numpy as np
import DataUtils

import masking_utils

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


print(torch.__version__)


print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
print('running device:', device)

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
model = AutoModelForMaskedLM.from_pretrained("klue/roberta-base")
model.to(device)
model.eval()

print('model setting...')
aug = DataUtils.Augmentation('./corpus/*' + '' + '*/**/*.json')

print('korlex setting...')

dirlist = ['도서_result_file.txt']

select_index = 0

for file_name in dirlist:
    file = open(file_name, 'r', encoding='utf-8')
    # new_file = open('기계독해_adversarial_QA', 'w', encoding='utf-8')

    print('file read...')

    data_str = file.read()

    whole_data = data_str.split('\t\n\n\t')
    whole_data.pop(-1)

    data_num = int(len(whole_data) / 4)
    whole_data = whole_data[select_index * data_num: (select_index + 1) * data_num]

    count = 0
    false_count = 0

    max_length = 512
    input_ids = np.zeros(shape=[len(whole_data) * 5, max_length], dtype=np.int32)
    token_type_ids = np.zeros(shape=[len(whole_data) * 5, max_length], dtype=np.int32)
    answer_span = np.zeros(shape=[len(whole_data) * 5, 2], dtype=np.int32)

    for d, data in enumerate(whole_data):
        lines = data.split('\n')
        query_sibling_data = lines[0]
        query_sibling_data2 = ''
        answer_sibling_data = lines[1]
        question = lines[2]
        context = lines[3]
        answer_text = lines[4]
        answer_start = int(lines[5])

        question_to_transferred = "" + question

        try:
            if len(query_sibling_data.strip()) > 0:
                tks = query_sibling_data.split('\t')
                words_to_replaced = tks.pop(0)
                siblings = tks

                sibling_word = masking_utils.get_sibling_prediction(
                    question=question_to_transferred,
                    text_to_replaced=words_to_replaced,
                    siblings=siblings,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )

                question_to_transferred = aug.conversion_josa(
                    question_to_transferred,
                    words_to_replaced,
                    sibling_word
                )

            if len(query_sibling_data2.strip()) > 0:
                tks = query_sibling_data2.split('\t')
                words_to_replaced = tks.pop(0)
                siblings = tks

                sibling_word = masking_utils.get_sibling_prediction(
                    question=question_to_transferred,
                    text_to_replaced=words_to_replaced,
                    siblings=siblings,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )

                question_to_transferred = aug.conversion_josa(
                    question_to_transferred,
                    words_to_replaced,
                    sibling_word
                )

            question_to_transferred = aug.transfer_sentence_to_plain(
                question=question_to_transferred,
                answer=answer_text,
                change_answer=answer_text
            )

            if len(answer_sibling_data.strip()) > 0:
                tks = answer_sibling_data.split('\t')
                words_to_replaced = tks.pop(0)
                siblings = tks

                sibling_word = masking_utils.get_sibling_prediction(
                    question=question_to_transferred,
                    text_to_replaced=words_to_replaced,
                    siblings=siblings,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )

                question_to_transferred = aug.conversion_josa(
                    question_to_transferred,
                    words_to_replaced,
                    sibling_word
                )
        except:
            print('continued')
            false_count += 1
            continue

        ans1 = '[STA]'
        context = context[0: answer_start] + ans1 + answer_text + context[answer_start + len(answer_text): -1]

        paragraphs = str(context).replace('. ', '. \n').split('\n')

        insert_idx = randint(0, len(paragraphs))
        paragraphs.insert(insert_idx, question_to_transferred)

        new_doc = ''

        for paragraph in paragraphs:
            new_doc += paragraph + ' '

        ###############
        paragraph = new_doc
        paragraph = paragraph.replace('[END]', '')
        answer_start = paragraph.find('[STA]')
        paragraph = paragraph.replace('[STA]', '')

        query_text = ''

        try:
            tokens, start_position, end_position = read_squad_example(orig_answer_text=answer_text,
                                                                      paragraph_text=paragraph,
                                                                      answer_offset=answer_start)

            input_ids_arrays, input_segments_arrays, start_positions, end_positions, doc_tokens = \
                convert_example_to_tokens(
                    question_text=question,
                    start_position=start_position,
                    end_position=end_position,
                    doc_tokens=tokens,
                    orig_answer_text=answer_text,
                    answer_text=answer_text)

            for s in range(len(input_ids_arrays)):
                input_ids_array = input_ids_arrays[s]
                input_segments_array = input_segments_arrays[s]
                start_position = start_positions[s]
                end_position = end_positions[s]
                doc_token = doc_tokens[s]

                if start_position == 0 or end_position == 0:
                    continue

                input_ids[count] = input_ids_array
                token_type_ids[count] = input_segments_array
                answer_span[count, 0] = start_position
                answer_span[count, 1] = end_position
                count += 1

                """
                print(doc_token)
                print(doc_token[start_position: end_position + 1])
                print(start_position, end_position)
                print(answer_text)
                print(question)
                print('---------------------------------------------------------')
                """
                if count % 10 == 0:
                    print(count, d, '/', len(whole_data), 'false count:', false_count)
        except:
            print('continued!')

    np.save(f'input_ids_{file_name}' + str(select_index), input_ids[0:count])
    np.save(f'segment_ids_{file_name}' + str(select_index), token_type_ids[0:count])
    np.save(f'answer_span_{file_name}' + str(select_index), answer_span[0:count])
