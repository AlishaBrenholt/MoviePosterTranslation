import torch
import torch.nn as nn
import json
from transformers import MarianTokenizer
import sentencepiece as sp
from torch.utils.data import Dataset, DataLoader
import time
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


# Note I took a lot of inspiration from https://medium.com/@WamiqRaza/sequence-to-sequence-learning-with-neural-networks-30028d824591
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.rnn(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        prediction = self.linear(output)
        return prediction, hidden, cell


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, english, korean):
        hidden, cell = self.encoder(english)
        prediction, _, _ = self.decoder(korean, hidden, cell)
        return prediction

class FullLLM():
    def __init__(self):
        SEED = 14
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.DEV_PATH = "data/llm/development.json"
        self.EVAL_PATH = "data/llm/evaluation.json"

        # this assumes that the tokenizers are already trained
        self.english_tokenizer = sp.SentencePieceProcessor()
        self.english_tokenizer.load("models/tokens/english.model")
        self.korean_tokenizer = sp.SentencePieceProcessor()
        self.korean_tokenizer.load("models/tokens/korean.model")

        self.english_vocab, self.reverse_english_vocab = self.load_sentencepiece_vocab("models/tokens/english.model")
        self.korean_vocab, self.reverse_korean_vocab = self.load_sentencepiece_vocab("models/tokens/korean.model")

        self.embedding_size = 256
        self.hidden_size = 512
        self.layers = 2
        self.encoder = Encoder(len(self.english_vocab), self.embedding_size, self.hidden_size, self.layers).to(self.device)
        self.decoder = Decoder(len(self.korean_vocab), self.embedding_size, self.hidden_size, self.layers).to(self.device)
        self.model = Model(self.encoder, self.decoder)
        self.model.load_state_dict(torch.load("checkpoints/llm/epoch_10.pt"))
        self.model.eval()


    def process_data(self):
        def get_data(path):
            with open(path, "r") as file:
                data = json.load(file)
            return data

        def make_plain_text(data, path):
            f = open(path + ".en", "w")
            g = open(path + ".ko", "w")
            for group in data:
                sentences = group["text"]
                for sentence in sentences:
                    f.write(sentence["en_text"] + "\n")
                    g.write(sentence["ko_text"] + "\n")
            f.close()
            g.close()

        # test = get_data(TEST_PATH)
        train_json = get_data(self.DEV_PATH)
        test_json = get_data(self.EVAL_PATH)
        make_plain_text(train_json, "data/llm/development")
        make_plain_text(test_json, "data/llm/evaluation")

        # bpe tokenization
        vocab_size = 2000
        threads = 16

        # this is used latter in batches, apparently it crashes if batches have different length sentences inside of themselves
        padding_id = 3

        sp.SentencePieceTrainer.train(input="data/llm/development.en", model_prefix="data/models/tokens/english",
                                      vocab_size=vocab_size, model_type="bpe", num_threads=threads, bos_id=0, eos_id=1,
                                      unk_id=2,
                                      pad_id=padding_id)  # force the padding id because it didn't automatically make it
        sp.SentencePieceTrainer.train(input="data/llm/development.ko", model_prefix="data/models/tokens/korean",
                                      vocab_size=vocab_size, model_type="bpe", num_threads=threads, bos_id=0, eos_id=1,
                                      unk_id=2,
                                      pad_id=padding_id)  # force the padding id because it didn't automatically make it
        english_tokenizer = sp.SentencePieceProcessor()
        english_tokenizer.load("data/models/tokens/english.model")
        korean_tokenizer = sp.SentencePieceProcessor()
        korean_tokenizer.load("data/models/tokens/korean.model")
        assert (english_tokenizer.pad_id() == 3)

    def load_sentencepiece_vocab(self,model_file):
        spp = sp.SentencePieceProcessor()
        spp.load(model_file)
        vocab = {}
        reverse_vocab = {}
        for i in range(spp.get_piece_size()):
            vocab[spp.id_to_piece(i)] = i
            reverse_vocab[i] = spp.id_to_piece(i)
        return vocab, reverse_vocab

    def translate(self, sentence, gen_length=20):
        with torch.no_grad():
            sentence = self.english_tokenizer.encode(sentence)
            sentence = torch.tensor(sentence).unsqueeze(0).to(self.device)
            hidden, cell = self.model.encoder(sentence)
            start = torch.tensor([[self.korean_tokenizer.bos_id()]]).to(self.device)
            output = []
            for _ in range(gen_length):
                prediction, hidden, cell = self.model.decoder(start, hidden, cell)
                prediction = prediction.argmax(2)
                output.append(prediction.item())
                start = prediction
                if prediction.item() == self.korean_tokenizer.eos_id():
                    break
            output = [self.reverse_korean_vocab[i] for i in output]
            return "".join(output)


if __name__ == "__main__":
    llm = FullLLM()
    print(llm.translate("Hello World"))
