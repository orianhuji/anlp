import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast 
import torch

from dataset_stats import datasets

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

def read_set(set):
    
    contexts = []
    questions = []
    answers = []

    for group in set:
        paragraphs = group['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def add_tags(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_tags = answer['answer_start']
        end_tags = answer['answer_end']

def add_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def bert_preprocessing(dataset_name):
    path = Path(f'../data/{dataset_name}.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    train, val = train_test_split(data['data'], test_size=0.2, shuffle=True)

    train_contexts, train_questions, train_answers = read_set(train)
    val_contexts, val_questions, val_answers = read_set(val)

    add_tags(train_answers, train_contexts)
    add_tags(val_answers, val_contexts)

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    add_positions(train_encodings, train_answers)
    add_positions(val_encodings, val_answers)

    return Dataset(train_encodings), Dataset(val_encodings)

def gpt_preprocessing(dataset_name):
    path = Path(f'../data/{dataset_name}.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    return train_test_split(data['data'], test_size=0.2, shuffle=True)

bert_datasets = {}
gpt_datasets = {}

for dataset_name in datasets.keys():
    bert_datasets[dataset_name] = {}
    gpt_datasets[dataset_name] = {}

    train_dataset, val_dataset = bert_preprocessing(dataset_name)
    bert_datasets[dataset_name]['train'] = train_dataset
    bert_datasets[dataset_name]['val'] = val_dataset
    
    train_dataset, val_dataset = gpt_preprocessing(dataset_name)
    gpt_datasets[dataset_name]['train'] = train_dataset
    gpt_datasets[dataset_name]['val'] = val_dataset
