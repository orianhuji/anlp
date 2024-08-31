# Transformer Models for Question Answering on ASD QA Dataset in Multilingual Setting

## Abstract:

Transformer-based question-answering (QA) models promise to enhance inclusive education by being tested and optimized with small, specific datasets before deployment. However,
sociomedical research indicates these models can be unpredictable. The original study indicates that while generative QA models may misrepresent facts and produce false tokens, 
they can enhance system output diversity and improve user-friendliness for younger users. It also suggests that, despite the higher reliability of extractive QA models, these 
models might be less efficient than their generative counterparts according to the metric scores.

Our investigation further explores these findings by reproducing the existing study, evaluating QA models on the Russian dataset, and extending the research by including 
English-translated data. We evaluate according to the original paper (Firsanova, 2022) both extractive and generative QA models, including BERT (Devlin et al., 2019), 
DistilBERT (Sanh et al., 2020), XLM-RoBERTa (Conneau et al., 2020), ruBERT (Zmitrovich et al., 2024), and GPT-2 (Radford et al., 2019), on a custom dataset of 1,134 question-answer 
pairs about autism spectrum disorders (ASD) in Russian. The dataset was translated to English using the Tower-instruct (Alves et al., 2024) model, followed by a human evaluation of the translated
content. An expert translated a subset of the dataset to ensure high-quality translations. The study tests the model’s performance on the expert-translated data compared to the 
machine-translated dataset.

------------------------------------------------------------------------------------------------------

## Code:

To reproduce the study for the extractive models (MBERT, DBERT, XLM, ruBERT), run the following scripts: 
https://github.com/orianhuji/anlp/blob/main/original_proj/BERT_based_QA/full_flow_mine_tranlation_alina.ipynb (based on the original dataset)

https://github.com/orianhuji/anlp/blob/main/original_proj/BERT_based_QA/full_flow_mine_tranlation.ipynb (based on Tower translated dataset)

https://github.com/orianhuji/anlp/blob/main/original_proj/BERT_based_QA/full_flow_mine_tranlation_alina.ipynb (based on the expert dataset)


To reproduce the study for the generative model (GPT-2), run the following scripts: 
https://github.com/orianhuji/anlp/blob/main/original_proj/GPT-2_based_QA/full_flow_mine_valid.ipynb (based on the original dataset)

https://github.com/orianhuji/anlp/blob/main/original_proj/GPT-2_based_QA/full_flow_mine_tranlation.ipynb (based on Tower translated dataset)

https://github.com/orianhuji/anlp/blob/main/original_proj/GPT-2_based_QA/full_flow_mine_tranlation_alina.ipynb(based on the expert dataset)


To translate the original dataset with Unlabel_TowerInstruct: https://github.com/orianhuji/anlp/blob/main/src/translate_russian_to_english.py
