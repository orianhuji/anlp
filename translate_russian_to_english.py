# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import json
import os.path

from tqdm import tqdm
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")

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
else:
    TRANSLATION_CACHE = {}
SPLIT_BY = '\nEnglish:<|im_end|>\n<|im_start|>assistant\n'
def fix_translation(in_text):
    return in_text[in_text.find(SPLIT_BY)+len(SPLIT_BY):]

def translate(input_txt):
    if TRANSLATION_CACHE.get(input_txt):
        return TRANSLATION_CACHE[input_txt]
    messages = [{
        "role": "user",
        "content": f"Translate the following text from Russian into English.\nRussian: {input_txt}\nEnglish:"
    }, ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
    TRANSLATION_CACHE[input_txt] = fix_translation(outputs[0]["generated_text"])
    with open('./data/translation_cache.json', 'w', encoding='utf-8') as translation_f:
        translation_f.write(json.dumps(TRANSLATION_CACHE, indent=4, ensure_ascii=False))
    return TRANSLATION_CACHE[input_txt]


with open('./data/original.json', 'r', encoding='utf-8') as original_ds_f:
    original_ds = json.loads(original_ds_f.read())

with open('./data/translated.json', 'r', encoding='utf-8') as translated_ds_f:
    translated_ds = json.loads(translated_ds_f.read())

invalid_counter = 0
for idx, record in tqdm(enumerate(original_ds['data'])):
    # record['title_eng'] = translate(record['title'])
    record['title_eng'] = translated_ds[idx]['title_eng']
    TRANSLATION_CACHE[record['title']] = record['title_eng']

    for idxx, paragraph in tqdm(enumerate(record['paragraphs'])):
        # paragraph['context_en'] = translate(paragraph['context'])
        paragraph['context_en'] = translated_ds[idx]['paragraphs'][idxx]['context_en']
        TRANSLATION_CACHE[paragraph['context']] = paragraph['context_en']

        for idxxx, qa in enumerate(paragraph['qas']):
            # qa['question_en'] = translate(qa['question'])
            qa['question_en'] = translated_ds[idx]['paragraphs'][idxx]['qas'][idxxx]['question_en']
            TRANSLATION_CACHE[qa['question']] = qa['question_en']

            for answer in qa['answers']:
                answer['text_en'] = translate(answer['text'])

                if not qa['is_impossible']:
                    answer['answer_start_en'] = paragraph['context_en'].find(answer['text_en'])
                    if answer['answer_start_en'] != -1:
                        answer['answer_end_en'] = answer['answer_start_en'] + len(answer['text_en'])
                    else:
                        invalid_counter += 1
with open('./data/translation_cache.json', 'w', encoding='utf-8') as translation_f:
    translation_f.write(json.dumps(TRANSLATION_CACHE, indent=4, ensure_ascii=False))
print(f'number of invalid translations => {invalid_counter}')

with open('./data/translated.json', 'w', encoding='utf-8') as translated_ds_f:
    translated_ds_f.write(json.dumps(original_ds['data'], indent=4, ensure_ascii=False))







