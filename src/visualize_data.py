import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

from dataset_stats import datasets

# Download stopwords if not already available
nltk.download('stopwords')
nltk.download('punkt')
russian_stopwords = set(stopwords.words('russian'))

# Load your dataset
df = pd.DataFrame({'question': datasets['short'][0], 'answer': datasets['short'][1]})

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in russian_stopwords]
    return ' '.join(tokens)

df['question'] = df['question'].astype(str).apply(preprocess_text)
df['answer'] = df['answer'].astype(str).apply(preprocess_text)

# Create a graph
G = nx.Graph()

# Add nodes
for i, row in df.iterrows():
    question_node = f"Q{i}"
    answer_node = f"A{i}"
    G.add_node(question_node, label=row['question'])
    G.add_node(answer_node, label=row['answer'])
    G.add_edge(question_node, answer_node)

# Create edges based on shared words
def add_shared_word_edges(G, df):
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i < j:
                shared_words = set(row1['question'].split()).intersection(set(row2['question'].split()))
                if shared_words:
                    G.add_edge(f"Q{i}", f"Q{j}", weight=len(shared_words))

add_shared_word_edges(G, df)

# Create edges based on semantic similarity using TF-IDF and cosine similarity
def add_similarity_edges(G, texts, threshold=0.2):
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if cosine_similarities[i, j] > threshold:
                G.add_edge(f"Q{i}", f"Q{j}", weight=cosine_similarities[i, j])

texts = df['question'].tolist()
add_similarity_edges(G, texts)

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, node_color="skyblue", font_color="black", font_weight="bold")
plt.title('Q/A Network Graph')
plt.show()