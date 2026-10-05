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
embedding_dim = 768

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

# Positional encoding visualization

plt.figure(figsize=(12, 6))

plt.imshow(positional_encoding, aspect="auto")

plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.title("Positional Encoding Pattern")

plt.colorbar(label="Encoding Value")

plt.show()