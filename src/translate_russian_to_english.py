import torch
import json
import os.path
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

TRANSLATION_MODEL = 'Unbabel/TowerInstruct-v0.1'
TRANSLATION_MODEL_CACHE = TRANSLATION_MODEL + '_substring_logic'
# TRANSLATION_MODEL = 'facebook/nllb-200-3.3B'

if 'nllb' in TRANSLATION_MODEL:
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

    pipe = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='rus_Cyrl', tgt_lang='eng_Latn',
                    max_length=1024, device_map="auto")
else:
    pipe = pipeline("text-generation", model=TRANSLATION_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

if os.path.exists('./original_proj/data/translation_cache.json'):
    with open('./original_proj/data/translation_cache.json', 'r', encoding='utf-8') as translation_f:
        TRANSLATION_CACHE = json.loads(translation_f.read() or '{}')
        TRANSLATION_CACHE[TRANSLATION_MODEL] = TRANSLATION_CACHE.get(TRANSLATION_MODEL, {})
        TRANSLATION_CACHE[TRANSLATION_MODEL_CACHE] = TRANSLATION_CACHE.get(TRANSLATION_MODEL_CACHE, {})
else:
    TRANSLATION_CACHE = {TRANSLATION_MODEL: {}, TRANSLATION_MODEL_CACHE: {}}

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
    with open('./original_proj/data/translation_cache.json', 'w', encoding='utf-8') as translation_f:
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


original_ds = pickle.loads(open("./original_proj/data/squad_dataset.pkl","rb").read())

for ds in ['train', 'validation']:
    invalid_counter = 0
    for record in tqdm(original_ds[ds]):

        record['answers']['text_en'] = [translate(t) for t in record['answers']['text']]
        record['context_en'] = ''
        last_idx = 0
        for i in range(len(record['answers']['text_en'])):
            record['context_en'] += translate(record['context'][last_idx:record['context'].find(record['answers']['text'][i])])
            record['context_en'] += ' ' + record['answers']['text_en'][i]
            last_idx = record['context'].find(record['answers']['text'][i]) + len(record['answers']['text_en'][i])

        record['context_en'] += translate(record['context'][last_idx:])
        record['question_en'] = translate(record['question'])
        
        record['answers']['answer_start_en'] = []
        for i in range(len(record['answers']['text'])):
            if record['answers']['text_en'][i] not in record['context_en']:
                invalid_counter += 1
                record['answers']['answer_start_en'].append(0)
            else:
                record['answers']['answer_start_en'].append(record['context_en'].find(record['answers']['text_en'][i]))
    print("ds", ds, "invalid_counter", invalid_counter)

with open(f'./original_proj/data/translated_{TRANSLATION_MODEL_CACHE.replace("/", "_")}.json', 'w', encoding='utf-8') as translated_ds_f:
    translated_ds_f.write(json.dumps(original_ds, indent=4, ensure_ascii=False))
