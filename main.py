from transformers import AutoTokenizer

# Tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love machine learning"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Text: ",text)
print("Tokens: ",tokens)
print("Token IDs: ",token_ids)
print("Decoded: ",tokenizer.decode(token_ids))