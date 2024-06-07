from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import string 

with open('vocab.txt', 'r') as f:
    vocab = {word.strip(): index for index, word in enumerate(f)}

app = Flask(__name__)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
model = TextClassificationModel(95811, 64, 4)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['GET'])
def predict():
    text = request.args.get('text').lower()
    # Remove punctuation
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    print(text)
    tokens = text.split()
    numerical_indices = [vocab[token] for token in tokens]
    text_tensor = torch.tensor(numerical_indices)
    offsets = torch.tensor([0])
    output = model(text_tensor, offsets)
    return jsonify({'output': output.tolist()})

if __name__ == '__main__':
    app.run(debug = True)

# You can try with the sentence: The government announced new measures to tackle climate change and promote renewable energy