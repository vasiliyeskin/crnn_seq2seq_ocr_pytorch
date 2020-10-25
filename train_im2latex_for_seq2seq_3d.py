import argparse
import random
import os
import pickle

import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data

import matplotlib.pyplot as plt

import tensorboardX.utils as xutils
import tensorboardX.x2num as x2num
import torchvision.utils as vutils


import src.utils as utils
import src.dataset as dataset

import crnn.seq2seq_3d_version as S2S

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, default='data/sample/train_filter.lst', help='path to train dataset list file')
parser.add_argument('--eval_list', type=str, default='data/sample/validate_filter.lst', help='path to evalation dataset list file')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading num_workers')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--img_width', type=int, default=280, help='the width of the input image to network')
# parser.add_argument('--img_height', type=int, default=160, help='the height of the input image to network')
# parser.add_argument('--img_width', type=int, default=500, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--model', default='./model/im2latex/', help='Where to store samples and models')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
cfg = parser.parse_args()
print(cfg)

clip = 1

# load alphabet
# with open('./data/char_std_5990.txt') as f:
with open('./data/sample/latex_vocab.txt') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    # alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2

# load list of formulas
with open('data/sample/formulas.norm.lst') as file:
    formulas = file.read().splitlines()

def tensor2image(x):
    tensor = x2num.make_np(vutils.make_grid(x.data[:64], normalize=True))
    xtensors = xutils.convert_to_HWC(tensor, 'CHW')
    plt.imshow(xtensors)
    plt.show()

def train(image, text, model, criterion, train_loader, teach_forcing_prob=1):
    # optimizer
    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    optimizer = torch.optim.Adam(model.parameters())

    # loss averager
    loss_avg = utils.Averager()

    for epoch in range(cfg.num_epochs):
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):

            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            # for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
            #     encoder_param.requires_grad = True
            #     decoder_param.requires_grad = True
            # encoder.train()
            # decoder.train()

            optimizer.zero_grad()

            # for model_param in zip(model.parameters()):
            #     model_param.requires_grad = True
            model.train()

            # formula = formulas(int(cpu_texts))
            target_variable = converter.encode(cpu_texts)
            utils.load_data(image, cpu_images)

            # # CNN + BiLSTM
            # encoder_outputs = encoder(image)
            if torch.cuda.is_available():
                target_variable = target_variable.cuda()

            output = model(image, target_variable)



            #     # start decoder for SOS_TOKEN
            #     decoder_input = target_variable[utils.SOS_TOKEN].cuda()
            #     decoder_hidden = decoder.initHidden(batch_size).cuda()
            # else:
            #     decoder_input = target_variable[utils.SOS_TOKEN]
            #     decoder_hidden = decoder.initHidden(batch_size)

            # # if i == 28:
            # # outputs for the test
            # print(f'    decoder_input {0}', decoder_input.shape)
            # print(f'    decoder_hidden{0}', decoder_hidden.shape)
            # print(f'    encoder_outputs{0}', encoder_outputs.shape)

            # tensor2image(cpu_images[0])
            # print(cpu_texts[0])
            # print(target_variable[0])

            loss = 0.0
            teach_forcing = True if random.random() > teach_forcing_prob else False
            # print('    teach_forcing: {}'.format(teach_forcing))
            # print('    decoder_input.shape[0] {}, batch_size {}, batch_size condition: {}'.format(decoder_input.shape[0], batch_size, decoder_input.shape[0] < batch_size))
            # if teach_forcing or decoder_input.shape[0] < cfg.batch_size:
            #     for di in range(1, target_variable.shape[0]):
            #
            #         # tensor2image(cpu_images[di])
            #         # print(cpu_texts[di])
            #         # print(target_variable[di])
            #         # print([converter.decode(item) for item in target_variable[di]])
            #
            #         decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #         decoder_output, decoder_hidden, decoder_attention = model(image)
            #         loss += criterion(decoder_output, target_variable[di])
            #         decoder_input = target_variable[di]
            # else:
            #     for di in range(1, target_variable.shape[0]):
            #         decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #         loss += criterion(decoder_output, target_variable[di])
            #         topv, topi = decoder_output.data.topk(1)
            #         ni = topi.squeeze()
            #         decoder_input = ni
            # encoder.zero_grad()
            # decoder.zero_grad()

            output_dim = output.shape[-1]
            # print(output_dim)

            output = output[1:].view(-1, output_dim)
            target_variable = target_variable[1:].view(-1)

            loss = criterion(output, target_variable)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            # encoder_optimizer.step()
            # decoder_optimizer.step()

            loss_avg.add(loss)

            if i % 1 == 0:
                print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch + 1, cfg.num_epochs, i + 1, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        # # save checkpoint
        # torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(cfg.model, epoch))
        # torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(cfg.model, epoch))


def evaluate(image, text, model, criterion, data_loader, max_eval_iter=100):

    # for e, d in zip(encoder.parameters(), decoder.parameters()):
    #     e.requires_grad = False
    #     d.requires_grad = False

    # encoder.eval()
    # decoder.eval()
    model.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    epoch_loss = 0

    with torch.no_grad():
        for i in range(min(len(data_loader), max_eval_iter)):
            cpu_images, cpu_texts = val_iter.next()
            batch_size = cpu_images.size(0)
            utils.load_data(image, cpu_images)

            target_variable = converter.encode(cpu_texts)
            n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        # encoder_outputs = encoder(image)
        if torch.cuda.is_available():
            target_variable = target_variable.cuda()
        #     decoder_input = target_variable[0].cuda()
        #     decoder_hidden = decoder.initHidden(batch_size).cuda()
        # else:
        #     decoder_input = target_variable[0]
        #     decoder_hidden = decoder.initHidden(batch_size)
        #
        # for di in range(1, target_variable.shape[0]):
        #     decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        #     topv, topi = decoder_output.data.topk(1)
        #     ni = topi.squeeze(1)
        #     decoder_input = ni
        #     if ni == utils.EOS_TOKEN:
        #         decoded_label.append(utils.EOS_TOKEN)
        #         break
        #     else:
        #         decoded_words.append(converter.decode(ni))
        #         decoded_label.append(ni)
            decoded_label = model(image, target_variable, 0)
            print(decoded_label.shape)
            images_number, batch, output_dim = decoded_label.size()
            decoded_label = decoded_label[1:].view(images_number, output_dim)
            target_variable = target_variable[1:].view(-1)
            print(decoded_label)
            print(target_variable)

            loss = criterion(decoded_label, target_variable)
            epoch_loss += loss.item()

            texts = cpu_texts[0]
            print(decoded_label.shape)
            decoded_words = [converter.decode(item) for item in decoded_label[0]]
            print('pred {}: {}'.format(i, ''.join(decoded_words)))
            print('gt {}: {}'.format(i, texts))

    accuracy = epoch_loss / max_eval_iter
    print('Test epoch loss: {}, accuray: {}'.format(epoch_loss, accuracy))


def get_formula(label):
    # function returnes the formula
    return formulas[int(label)]


def main():
    if not os.path.exists(cfg.model):
        os.makedirs(cfg.model)

    # path to images
    path_to_images = 'data/sample/images_processed/'

    # create train dataset
    train_dataset = dataset.TextLineDataset(text_line_file=cfg.train_list,
                                            transform=None,
                                            target_transform=get_formula,
                                            path_to_images=path_to_images)
    sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=sampler, num_workers=int(cfg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=cfg.img_height, img_width=cfg.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(text_line_file=cfg.eval_list,
                                           transform=dataset.ResizeNormalize(img_width=cfg.img_width, img_height=cfg.img_height),
                                           target_transform=get_formula,
                                           path_to_images=path_to_images
                                           )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=int(cfg.num_workers))

    # create input tensor
    image = torch.FloatTensor(cfg.batch_size, 3, cfg.img_height, cfg.img_width)
    text = torch.LongTensor(cfg.batch_size)

    # # create crnn/seq2seq/attention network
    # encoder = crnn.Encoder(channel_size=3, hidden_size=cfg.hidden_size)
    #
    # # max length for the decoder
    # max_width = cfg.max_width
    # max_width = encoder.get_max_lenght_for_Decoder(image)
    #
    # # for prediction of an indefinite long sequence
    # decoder = crnn.Decoder(hidden_size=cfg.hidden_size, output_size=num_classes, dropout_p=0.1, max_length=max_width)
    # print(encoder)
    # print(decoder)
    # encoder.apply(utils.weights_init)
    # decoder.apply(utils.weights_init)
    #
    #
    # if cfg.encoder:
    #     print('loading pretrained encoder model from %s' % cfg.encoder)
    #     encoder.load_state_dict(torch.load(cfg.encoder))
    # if cfg.decoder:
    #     print('loading pretrained encoder model from %s' % cfg.decoder)
    #     decoder.load_state_dict(torch.load(cfg.decoder))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = 512
    OUTPUT_DIM = num_classes
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    cnn = S2S.CNN(channel_size=3)
    attn = S2S.Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = S2S.Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = S2S.Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = S2S.Seq2Seq(cnn, enc, dec, device).to(device)

    model.apply(S2S.init_weights)
    print(model)

    print(f'The model has {S2S.count_parameters(model):,} trainable parameters')


    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    # assert torch.cuda.is_available(), "Please run \'train.py\' script on nvidia cuda devices."
    if torch.cuda.is_available():
        # encoder.cuda()
        # decoder.cuda()
        image = image.cuda()
        text = text.cuda()
        criterion = criterion.cuda()

    # train crnn
    train(image, text, model, criterion, train_loader, teach_forcing_prob=cfg.teaching_forcing_prob)

    # do evaluation after training
    evaluate(image, text, model, criterion, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    main()
