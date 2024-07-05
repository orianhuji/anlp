import json
from pathlib import Path

import statistics


def listing(data, list_q, list_a):
  for item in data['data']:
    for paragraph in item['paragraphs']:
      for qa in paragraph['qas']:
        list_q.append(qa['question'])
        list_a.append(qa['answers'][0]['text'])
  
  return list_q, list_a

def avg_symbols(strings):
  return round(sum(map(len, strings)) / len(strings))

def avg_tokens(strings):
  return round(statistics.mean([len(token) for token in [element.split() for element in strings]]))

def extract_jsons(dictionary):
  for dataset in dictionary.keys():
    path = Path(f'../data/{dataset}.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    listing(data, *dictionary[dataset][:])

def print_statistics(dictionary):
  for dataset in dictionary.keys():
    print(f'Average number of symbols in questions from {dataset} dataset version',
          avg_tokens(dictionary[dataset][0]))
    print(f'Average number of symbols in answers from {dataset} dataset version',
          avg_tokens(dictionary[dataset][1]))

datasets = {'original': ([], []), 
            'short': ([], []), 
            'translated': ([], []), 
            'translated_manually': ([], [])}

extract_jsons(datasets)
print_statistics(datasets)
