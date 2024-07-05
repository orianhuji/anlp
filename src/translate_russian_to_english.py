import torch
import json
import os.path

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

TRANSLATION_MODEL = 'Unbabel/TowerInstruct-v0.1'
TRANSLATION_MODEL_CACHE = TRANSLATION_MODEL + '_substring_logic'
# TRANSLATION_MODEL = 'facebook/nllb-200-3.3B'

if 'nllb' in TRANSLATION_MODEL:
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

    pipe = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='rus_Cyrl', tgt_lang='eng_Latn',
                    max_length=512, device_map="auto")
else:
    pipe = pipeline("text-generation", model=TRANSLATION_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

if os.path.exists('../data/translation_cache.json'):
    with open('../data/translation_cache.json', 'r', encoding='utf-8') as translation_f:
        TRANSLATION_CACHE = json.loads(translation_f.read() or '{}')
        TRANSLATION_CACHE[TRANSLATION_MODEL] = TRANSLATION_CACHE.get(TRANSLATION_MODEL, {})
        TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE] = TRANSLATION_CACHE.get(TRANSLATION_MODEL_CACHE, {})
else:
    TRANSLATION_CACHE = {TRANSLATION_MODEL: {}}

SPLIT_BY = '\nEnglish:<|im_end|>\n<|im_start|>assistant\n'


def fix_translation(in_text):
    return in_text[in_text.find(SPLIT_BY) + len(SPLIT_BY):]


def translate(input_txt):
    if TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE].get(input_txt):
        return TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE][input_txt]
    if not input_txt:
        return input_txt

    # Prefer replacing long sequence over small
    for russian_source in sorted(TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE].keys(), key=lambda k: len(k), reverse=True):
        # if russian_source in out_text:
        out_text = input_txt.replace(russian_source, TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE][russian_source])
    if 'nllb' in TRANSLATION_MODEL_CACHE:
        outputs = _nllb_translate(input_txt)
    else:
        outputs = _tower_translate(input_txt)
        outputs = fix_translation(outputs[0]["generated_text"])

    TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE][input_txt] = outputs
    with open('../data/translation_cache.json', 'w', encoding='utf-8') as translation_f:
        translation_f.write(json.dumps(TRANSLATION_CACHE, indent=4, ensure_ascii=False))
    return TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE][input_txt]


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


with open('../data/original.json', 'r', encoding='utf-8') as original_ds_f:
    original_ds = json.loads(original_ds_f.read())

invalid_counter = 0
for record in tqdm(original_ds['data']):
    record['title_eng'] = translate(record['title'])

    for paragraph in tqdm(record['paragraphs']):
        # translate short sequences and only then long ones, in order to allow (almost) maximum replacements
        answers_to_translate = [an['text'] for qa in paragraph['qas'] for an in qa['answers']]
        answers_to_translate = sorted(answers_to_translate, key=lambda x: len(x))
        for answer_to_translate in answers_to_translate:
            translate(answer_to_translate)

        for qa in paragraph['qas']:
            qa['question_en'] = translate(qa['question'])
            for answer in qa['answers']:
                answer['text_en'] = translate(answer['text'])
        paragraph['context_en'] = translate(paragraph['context'])

        for qa in paragraph['qas']:
            for answer in qa['answers']:
                if not qa['is_impossible']:
                    answer['answer_start_en'] = paragraph['context_en'].find(answer['text_en'])
                    if answer['answer_start_en'] != -1:
                        answer['answer_end_en'] = answer['answer_start_en'] + len(answer['text_en'])
                    else:
                        invalid_counter += 1

print(f'number of invalid translations => {invalid_counter}')
original_ds['invalid_cnt'] = invalid_counter

with open(f'../data/translated_{TRANSLATION_MODEL_CACHE.replace("/", "_")}.json', 'w', encoding='utf-8') as translated_ds_f:
    translated_ds_f.write(json.dumps(original_ds, indent=4, ensure_ascii=False))
