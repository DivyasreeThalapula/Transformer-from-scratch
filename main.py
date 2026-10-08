from transformers import AutoTokenizer
import numpy as np

np.random.seed(42)

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

def transformer_block(final_embeddings):
    # multi-head attention

    num_heads = 2
    head_dim = embedding_dim // num_heads

    heads = []

    # softmax

    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    for head in range(num_heads):

        # Query, Key, Value

        W_Q = np.random.randn(embedding_dim, head_dim)
        W_K = np.random.randn(embedding_dim, head_dim)
        W_V = np.random.randn(embedding_dim, head_dim)

        Q = final_embeddings @ W_Q
        K = final_embeddings @ W_K
        V = final_embeddings @ W_V

        # Attention scores

        scores = Q @ K.T

        # Scaling attention scores

        d_k = head_dim

        scaled_scores = scores / np.sqrt(d_k)

        # Attention weights

        attention_weights = softmax(scaled_scores)

        # Final attention output

        attention_output = attention_weights @ V

        heads.append(attention_output)

    # multi-head output

    multi_head_output = np.concatenate(heads, axis=-1)

    print("Multi-head output shape: ",multi_head_output.shape)
    print(multi_head_output)

    # output projection

    W_O = np.random.randn(embedding_dim, embedding_dim)

    projected_output = multi_head_output @ W_O

    print("Projected Output Shape", projected_output.shape)

    # residual connection around attention

    attention_residual = final_embeddings + projected_output
    print("Attention residual shape: ",attention_residual.shape)

    # Feed-forward network

    ffn_dim = embedding_dim * 2

    W1 = np.random.randn(embedding_dim, ffn_dim)
    b1 = np.zeros(ffn_dim)

    W2 = np.random.randn(ffn_dim, embedding_dim)
    b2 = np.zeros(embedding_dim)

    hidden = attention_residual @ W1 + b1
    print("FFN hidden shape: ", hidden.shape)

    hidden = np.maximum(hidden,0)
    print("After ReLU shape: ", hidden.shape)

    ffn_output = hidden @ W2 + b2
    print("FFN output shape: ", ffn_output.shape)

    # residual connection around ffn

    ffn_residual = attention_residual + ffn_output
    print("FFN residual shape: ", ffn_residual.shape)

    # Layer Normalization

    epsilon = 1e-5

    mean = np.mean(ffn_residual, axis=1, keepdims=True)
    variance = np.var(ffn_residual, axis=1, keepdims=True)

    normalized = (ffn_residual - mean ) / np.sqrt(variance + epsilon)

    gamma = np.ones(embedding_dim)
    beta = np.zeros(embedding_dim)

    layer_norm_output = gamma * normalized + beta

    print("Layer normalization output shape: ", layer_norm_output.shape)

    return layer_norm_output

num_blocks = 2

output = final_embeddings

for block in range(num_blocks):
    output = transformer_block(output)

print("Final output shape: ",output.shape)