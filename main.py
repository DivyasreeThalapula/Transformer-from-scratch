from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

# Tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love machine learning"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens: ",tokens)
print("Token IDs: ",token_ids)

# Embedding matrix

vocab_size = tokenizer.vocab_size
embedding_dim = 4

embedding_matrix = np.random.randn( vocab_size, embedding_dim) * 0.02

print("Embedding Matrix Shape: ",embedding_matrix.shape)

token_embeddings = embedding_matrix[token_ids]

print("Token Embeddings Shape: ",token_embeddings.shape)

# Positional encoding

def get_positional_encoding(seq_len, d_model):

    position = np.arange(seq_len)[:, np.newaxis]

    div_term = np.exp(
        np.arange(0, d_model, 2)
        * -(np.log(10000.0) / d_model)
    )

    positional_encoding = np.zeros((seq_len, d_model))

    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)

    return positional_encoding

# generate positional encoding

seq_len = len(token_ids)

positional_encoding = get_positional_encoding(seq_len, embedding_dim)

print("Positional Encoding Shape: ", positional_encoding.shape)

# Final embedding = Token embedding + Positional Encoding

final_embeddings = token_embeddings + positional_encoding

print("Final Embeddings Shape: ", final_embeddings.shape)

# Query, Key, Value

W_Q = np.random.randn(embedding_dim, embedding_dim)
W_K = np.random.randn(embedding_dim, embedding_dim)
W_V = np.random.randn(embedding_dim, embedding_dim)

Q = final_embeddings @ W_Q
K = final_embeddings @ W_K
V = final_embeddings @ W_V

print("Q shape: ", Q.shape)
print("K shape: ", K.shape)
print("V shape: ", V.shape)

# Attention scores

scores = Q @ K.T

print("Scores shape: ",scores.shape)
print(scores)

# Scaling attention scores

d_k = embedding_dim

scaled_scores = scores / np.sqrt(d_k)

print(scaled_scores)

# softmax

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Attention weights

attention_weights = softmax(scaled_scores)

print("Attention weights shape: ",attention_weights.shape)
print(attention_weights)

# Final attention output

attention_output = attention_weights @ V

print("Attention output shape: ",attention_output.shape)
print(attention_output)