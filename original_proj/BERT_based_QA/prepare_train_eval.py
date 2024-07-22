import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast 
import torch 
from transformers import BertForQuestionAnswering
import gc 
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import pickle
import pandas as pd
from tqdm import tqdm
import os


path = Path('../data/original.json')
data = json.loads(path.read_text(encoding='utf-8'))
df = pd.DataFrame(data)

df1 = pd.DataFrame(df['data'].values.tolist())
df1.columns = df1.columns
col = df.columns.difference(['data'])
df = pd.concat([df[col], df1],axis=1)

data = df.explode('paragraphs')['paragraphs'].to_list()

train, temp = train_test_split(data, test_size=0.3, shuffle=True)
val, test = train_test_split(temp, test_size=0.5, shuffle=True)

def read_set(set):
    
    contexts = []
    questions = []
    answers = []

    for group in set:
        context = group['context']
        for qa in group['qas']:
            question = qa['question']
            for answer in qa['answers']:
                contexts.append(context)
                questions.append(question)
                answers.append(answer)

    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_set(train)
val_contexts, val_questions, val_answers = read_set(val)

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
        return {key: torch.tensor(val[idx] or -1, dtype=torch.int64) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

for model_name, model_path in [
    ('m-bert', 'bert-base-multilingual-cased'),
    ('m-distil-bert', 'distilbert/distilbert-base-multilingual-cased'),
    ('ru-bert', 'DeepPavlov/rubert-base-cased'),
    ('xlm-roberta', 'FacebookAI/xlm-roberta-base'),
]:
    print('working on', model_name)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    add_positions(train_encodings, train_answers)
    add_positions(val_encodings, val_answers)


    train_dataset = Dataset(train_encodings)
    val_dataset = Dataset(val_encodings)

    os.makedirs(f"../data/{model_name}", exist_ok=True)

    with open(f"../data/{model_name}/train_dataset.pkl","wb") as file:
        pickle.dump(train_dataset, file)

    with open(f"../data/{model_name}/val_dataset.pkl","wb") as file:
        pickle.dump(val_dataset, file)

    with open(f"../data/{model_name}/train_answers.pkl","wb") as file:
        pickle.dump(val_answers, file)

    with open(f"../data/{model_name}/train_questions.pkl","wb") as file:
        pickle.dump(val_questions, file)

    with open(f"../data/{model_name}/train_contexts.pkl","wb") as file:
        pickle.dump(val_contexts, file)

    with open(f"../data/{model_name}/val_answers.pkl","wb") as file:
        pickle.dump(val_answers, file)

    with open(f"../data/{model_name}/val_questions.pkl","wb") as file:
        pickle.dump(val_questions, file)

    with open(f"../data/{model_name}/val_contexts.pkl","wb") as file:
        pickle.dump(val_contexts, file)


    model = BertForQuestionAnswering.from_pretrained(model_path)

    with open(f"../data/{model_name}/train_dataset.pkl","rb") as file:
        train_dataset = pickle.load(file)

    with open(f"../data/{model_name}/val_dataset.pkl","rb") as file:
        val_dataset = pickle.load(file)

    gc.collect() # used to prevent the "cuda running out of memory" error

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # model to GPU

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # batch size is 1 (because the model is large), the data is shuffled

    optim = AdamW(model.parameters(), lr=5e-5) # AdamW optimization algorithm, learning rate is 5e-5

    for epoch in tqdm(range(10)): # 10 epochs
        for batch in train_loader:
            optim.zero_grad() 
            input_ids = batch['input_ids'].to(device) # integers
            attention_mask = batch['attention_mask'].to(device) # 0's and 1's sequences
            start_positions = batch['start_positions'].to(device) # span
            end_positions = batch['end_positions'].to(device) 
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0] 
            loss.backward() # backpropagation 
            optim.step() # gradient descent

    filepath = f'../models/{model_name}.pth' # saving weights
    torch.save(model.state_dict(), filepath)
    model.load_state_dict(torch.load(filepath))
    model.eval() # model summary