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

input_ids = token_ids[:-1]
target_ids = token_ids[1:]

print("Input ids: ",input_ids)
print("Target ids: ",target_ids)

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

# softmax

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# sequence loss function

def sequence_loss(probabilities, target_ids):
    tot_loss = 0.0

    for position in range(len(target_ids)):

        correct_token_ids = target_ids[position]

        correct_probability = probabilities[ position, correct_token_ids]

        loss = -np.log(correct_probability + 1e-9)
        tot_loss += loss

    avg_loss = tot_loss / len(target_ids)

    return avg_loss

num_heads = 2
head_dim = embedding_dim // num_heads

W_Qs = [
    np.random.randn(embedding_dim, head_dim)
    for _ in range(num_heads)
]

W_Ks = [
    np.random.randn(embedding_dim, head_dim)
    for _ in range(num_heads)
]

W_Vs = [
    np.random.randn(embedding_dim, head_dim)
    for _ in range(num_heads)
]

W_O = np.random.randn(embedding_dim, embedding_dim)

ffn_dim = embedding_dim * 2

W1 = np.random.randn(embedding_dim, ffn_dim)
b1 = np.zeros(ffn_dim)

W2 = np.random.randn(ffn_dim, embedding_dim)
b2 = np.zeros(embedding_dim)

gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

# reusable transformer block

def transformer_block(final_embeddings):
    # multi-head attention

    head_dim = embedding_dim // num_heads

    heads = []

    for head in range(num_heads):

        # Query, Key, Value

        W_Q = W_Qs[head]
        W_K = W_Ks[head]
        W_V = W_Vs[head]

        Q = final_embeddings @ W_Q
        K = final_embeddings @ W_K
        V = final_embeddings @ W_V

        # Attention scores

        scores = Q @ K.T

        # Scaling attention scores

        d_k = head_dim

        scaled_scores = scores / np.sqrt(d_k)

        # casual/masked self-attention

        current_seq_len = final_embeddings.shape[0]
        causal_mask = np.tril(np.ones((current_seq_len, current_seq_len)))

        scaled_scores = np.where(
            casual_mask == 1,
            scaled_scores,
            -1e9
        )

        # Attention weights

        attention_weights = softmax(scaled_scores)

        # Final attention output

        attention_output = attention_weights @ V

        heads.append(attention_output)

    # multi-head output

    multi_head_output = np.concatenate(heads, axis=-1)

    # output projection

    projected_output = multi_head_output @ W_O

    # residual connection around attention

    attention_residual = final_embeddings + projected_output

    # Feed-forward network

    hidden = attention_residual @ W1 + b1

    hidden = np.maximum(hidden,0)

    ffn_output = hidden @ W2 + b2

    # residual connection around ffn

    ffn_residual = attention_residual + ffn_output

    # Layer Normalization

    epsilon = 1e-5

    mean = np.mean(ffn_residual, axis=1, keepdims=True)
    variance = np.var(ffn_residual, axis=1, keepdims=True)

    normalized = (ffn_residual - mean ) / np.sqrt(variance + epsilon)

    layer_norm_output = gamma * normalized + beta

    return layer_norm_output

num_blocks = 2

output = final_embeddings

for block in range(num_blocks):
    output = transformer_block(output)

# output projection to vocabulary

W_vocab = np.random.randn(embedding_dim, vocab_size)

# training output

training_output = output[:len(target_ids)]

# training loop

learning_rate = 0.01
num_steps = 3000

for step in range(num_steps):

    # forward pass
    logits = training_output @ W_vocab
    probabilities = softmax(logits)

    # calculate loss
    loss = sequence_loss(probabilities, target_ids)

    # calculate gradient
    gradient_logits = probabilities.copy()

    for position in range(len(target_ids)):
        correct_token_id = target_ids[position]
        gradient_logits[position, correct_token_id] -= 1

    gradient_logits /= len(target_ids)

    gradient_W_vocab = training_output.T @ gradient_logits

    # update weights using gradient descent
    W_vocab -= learning_rate * gradient_W_vocab

    # training progess after every 100 steps
    if step % 100 == 0:
        print("Step:", step, "Loss:", loss)

# loss after training

final_logits = training_output @ W_vocab
final_probabilities = softmax(final_logits)

final_loss = sequence_loss(final_probabilities, target_ids)

print("Final loss:", final_loss)

# get predicted tokens

predicted_token_ids = np.argmax(final_probabilities, axis=1)
print("Predicted token IDs:", predicted_token_ids)

predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
print("Predicted tokens:", predicted_tokens)

# compare predictions with correct tokens

correct_predictions = np.sum(
    predicted_token_ids == target_ids
)

total_predictions = len(target_ids)

print("Correct predictions:", correct_predictions)
print("Total predictions:", total_predictions)

accuracy = correct_predictions / total_predictions

print("Training accuracy:", accuracy * 100, "%")