from transformers import AutoTokenizer
import numpy as np

# Tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love machine learning"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens: ",tokens)
print("Token IDs: ",token_ids)

# Embedding matrix

vocab_size = 50000
embedding_dim = 768

embedding_matrix = np.random.randn( vocab_size, embedding_dim) * 0.02

print("Embedding Matrix Shape: ",embedding_matrix.shape)

token_embeddings = embedding_matrix[token_ids]

print("Token Embeddings Shape: ",token_embeddings.shape)
print(token_embeddings)