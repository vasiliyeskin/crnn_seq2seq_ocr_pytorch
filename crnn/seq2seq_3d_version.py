# based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/3 - Neural Machine Translation by Jointly Learning to Align and Translate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
# from torchtext.data.metrics import bleu_score

import spacy
import numpy as np

import random
import math
import time
import pandas as pd


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_text(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in text.split(' ')]

class CNN(nn.Module):

    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
                      nn.Conv2d(channel_size, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))

    def forward(self, input):
        # [n, channel_size, 32, 280] -> [n, 512, 1, 71]
        conv = self.cnn(input)
        return conv

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        # self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        # embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(src)

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.hidden_size = dec_hid_dim

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)

    def initHidden(self, batch_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

class Seq2Seq(nn.Module):
    def __init__(self, cnn, encoder, decoder, device):
        super().__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, image, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = image.shape[0]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # print(trg_vocab_size)

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # image to the convolution neural network
        cnn_outputs = self.cnn(image)
        b, c, h, w = cnn_outputs.size()
        cnn_outputs = cnn_outputs.view(b, c, h * w)  # [b, c, h, w] -> [b, c, h*w]
        cnn_outputs = cnn_outputs.permute(2, 0, 1)  # [b, c, w] -> [w, b, c]


        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(cnn_outputs)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # print(output.shape)
            # print(outputs.shape)
            # print(trg.shape)
            # print(batch_size)
            # print(image.shape)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]

# def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
#     trgs = []
#     pred_trgs = []
#
#     for datum in data:
#         src = vars(datum)['src']
#         trg = vars(datum)['trg']
#
#         pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)
#
#         # cut off <eos> token
#         pred_trg = pred_trg[:-1]
#
#         pred_trgs.append(pred_trg)
#         trgs.append([trg])
#
#     return bleu_score(pred_trgs, trgs)

def from_txt_to_dataframe_and_csv(dir, src_file, tgt_file, file_csv):
    src_text = open(f'{dir}/{src_file}').read().split('\n')
    tgt_text = open(f'{dir}/{tgt_file}').read().split('\n')
    raw_data = {'src': [line for line in src_text], 'tgt': [line for line in tgt_text]}
    df = pd.DataFrame(raw_data, columns=["src", "tgt"])
    df.to_csv(f"{dir}/{file_csv}.csv", index=False)


if __name__ == "__main__":
    # spacy_de = spacy.load('de')
    # spacy_en = spacy.load('en')
    #
    # SRC = Field(tokenize=tokenize_de,
    #             init_token='<sos>',
    #             eos_token='<eos>',
    #             lower=True)
    #
    # TRG = Field(tokenize=tokenize_en,
    #             init_token='<sos>',
    #             eos_token='<eos>',
    #             lower=True)
    #
    # train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
    #                                                     fields=(SRC, TRG))
    #
    # SRC.build_vocab(train_data, min_freq=2)
    # TRG.build_vocab(train_data, min_freq=2)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # BATCH_SIZE = 128
    #
    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    #     batch_size=BATCH_SIZE,
    #     device=device)

    #######################################
    ####### test with invert ##############
    #######################################
    SRC_TRN_PATH, TRG_TRN_PATH = 'toy-revert/src-train.txt',  'toy-revert/tgt-train.txt'
    SRC_VAL_PATH, TRG_VAL_PATH = 'toy-revert/src-val.txt', 'toy-revert/tgt-val.txt'
    SRC_TEST_PATH, TRG_TEST_PATH = 'toy-revert/src-test.txt', 'toy-revert/tgt-test.txt'

    TEXT = Field(tokenize="spacy",
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True)

    TRG_TEXT = Field(tokenize="spacy",
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True)

    from_txt_to_dataframe_and_csv('toy-revert', 'src-train.txt', 'trg-train.txt', 'train')
    from_txt_to_dataframe_and_csv('toy-revert', 'src-val.txt', 'trg-val.txt', 'val')
    from_txt_to_dataframe_and_csv('toy-revert', 'src-test.txt', 'trg-test.txt', 'test')

    data_fields = [('src', TEXT), ('trg', TRG_TEXT)]
    # load the dataset in csv format
    train_data, valid_data, test_data = TabularDataset.splits(
        path='toy-revert',
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        fields=data_fields,
        skip_header=True
    )

    TEXT.build_vocab(train_data)
    TRG_TEXT.build_vocab(train_data)
    SRC, TRG = TEXT, TRG_TEXT


    BATCH_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    #######################################

    INPUT_DIM = 512
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    cnn = CNN(channel_size=3)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(cnn, enc, dec, device).to(device)

    model.apply(init_weights)
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut3-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load('tut3-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')



    ##########################################################
    ##### testing with the life example
    example_idx = 10

    src = vars(train_data.examples[example_idx])['src']
    trg = vars(train_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')

    src = ['a', 'b', 'c', 'a', 'd']
    print(src)
    translation = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')

    src = 'd b c d .'.split(' ')
    print(src)
    translation = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')


    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
    print(f'BLEU score = {bleu_score * 100:.2f}')






