import json
from pathlib import Path

import statistics

def paragraph_listing(data, list_p, is_translated):
  for item in data['data']:
    for paragraph in item['paragraphs']:
      if is_translated and 'context_en' in paragraph.keys():
        list_p.append(paragraph['context_en'])
      elif not is_translated:
        list_p.append(paragraph['context'])

def qa_listing(data, list_q, list_a, is_translated):
  for item in data['data']:
    for paragraph in item['paragraphs']:
      for qa in paragraph['qas']:
        if is_translated and 'question_en' in qa.keys():
          list_q.append(qa['question_en'])
        elif not is_translated:
          list_q.append(qa['question'])

        if is_translated and 'text_en' in qa['answers'][0].keys():
          list_a.append(qa['answers'][0]['text_en'])
        elif not is_translated:
          list_a.append(qa['answers'][0]['text'])
  
  return list_q, list_a

def max_symbols(strings):
  return max(map(len, strings))

def avg_symbols(strings):
  return round(sum(map(len, strings)) / len(strings))

def min_tokens(strings):
  return min([len(tokens_list) for tokens_list in [element.split() for element in strings]])

def max_tokens(strings):
  return max([len(tokens_list) for tokens_list in [element.split() for element in strings]])

def avg_tokens(strings):
  return round(statistics.mean([len(tokens_list) for tokens_list in [element.split() for element in strings]]))

def extract_jsons(dictionary, is_translated):
  for dataset in dictionary.keys():
    path = Path(f'./data/{dataset}.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    paragraph_listing(data, dictionary[dataset][0], is_translated)
    qa_listing(data, *dictionary[dataset][1:], is_translated)

def print_stat(dictionary, stat, stat_name, metric):
  for dataset in dictionary.keys():
    print(f'{stat_name} number of {metric} in paragraphs from {dataset} dataset version',
         stat(dictionary[dataset][0]))
    print(f'{stat_name} number of {metric} in questions from {dataset} dataset version',
          stat(dictionary[dataset][1]))
    print(f'{stat_name} number of {metric} in answers from {dataset} dataset version',
          stat(dictionary[dataset][2]))

