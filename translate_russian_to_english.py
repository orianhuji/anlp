# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import json
import os.path

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# TRANSLATION_MODEL = 'Unbabel/TowerInstruct-v0.1'
#
# pipe = pipeline("text-generation", model=TRANSLATION_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

TRANSLATION_MODEL = 'facebook/nllb-200-distilled-600M'

model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

pipe = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='rus_Cyrl', tgt_lang='eng_Latn',
                max_length=512)

# We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating


# <|im_start|>user
# Translate the following text from Portuguese into English.
# Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
# English:<|im_end|>
# <|im_start|>assistant
# A group of researchers has launched a new model for translation-related tasks.

if os.path.exists('./data/translation_cache.json'):
    with open('./data/translation_cache.json', 'r', encoding='utf-8') as translation_f:
        TRANSLATION_CACHE = json.loads(translation_f.read() or '{}')
        TRANSLATION_CACHE[TRANSLATION_MODEL] = TRANSLATION_CACHE.get(TRANSLATION_MODEL, {})
else:
    TRANSLATION_CACHE = {TRANSLATION_MODEL: {}}

SPLIT_BY = '\nEnglish:<|im_end|>\n<|im_start|>assistant\n'


def fix_translation(in_text):
    return in_text[in_text.find(SPLIT_BY) + len(SPLIT_BY):]


def translate(input_txt):
    if TRANSLATION_CACHE[TRANSLATION_MODEL].get(input_txt):
        return TRANSLATION_CACHE[TRANSLATION_MODEL][input_txt]
    if 'nllb' in TRANSLATION_MODEL:
        outputs = _nllb_translate(input_txt)
    else:
        outputs = _tower_translate(input_txt)
        outputs = fix_translation(outputs[0]["generated_text"])

    TRANSLATION_CACHE[TRANSLATION_MODEL][input_txt] = outputs
    with open('./data/translation_cache.json', 'w', encoding='utf-8') as translation_f:
        translation_f.write(json.dumps(TRANSLATION_CACHE, indent=4, ensure_ascii=False))
    return TRANSLATION_CACHE[TRANSLATION_MODEL][input_txt]


def _nllb_translate(input_txt):
    output = pipe(input_txt)
    return output[0]['translation_text']


def _tower_translate(input_txt):
    messages = [{
        "role": "user",
        "content": f"Translate the following text from Russian into English.\nRussian: {input_txt}\nEnglish:"
    }, ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
    return outputs


with open('./data/original.json', 'r', encoding='utf-8') as original_ds_f:
    original_ds = json.loads(original_ds_f.read())

invalid_counter = 0
for record in tqdm(original_ds['data']):
    record['title_eng'] = translate(record['title'])

    for paragraph in tqdm(record['paragraphs']):
        paragraph['context_en'] = translate(paragraph['context'])

        for qa in paragraph['qas']:
            qa['question_en'] = translate(qa['question'])

            for answer in qa['answers']:
                answer['text_en'] = translate(answer['text'])

                if not qa['is_impossible']:
                    answer['answer_start_en'] = paragraph['context_en'].find(answer['text_en'])
                    if answer['answer_start_en'] != -1:
                        answer['answer_end_en'] = answer['answer_start_en'] + len(answer['text_en'])
                    else:
                        invalid_counter += 1

print(f'number of invalid translations => {invalid_counter}')
original_ds['invalid_cnt'] = invalid_counter

with open(f'./data/translated_{TRANSLATION_MODEL}.json', 'w', encoding='utf-8') as translated_ds_f:
    translated_ds_f.write(json.dumps(original_ds, indent=4, ensure_ascii=False))
