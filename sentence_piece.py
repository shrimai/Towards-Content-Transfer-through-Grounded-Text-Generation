###
# This code is based on the instructions given in https://github.com/google/sentencepiece/blob/master/python/README.md
###

import sentencepiece as spm
import argparse, codecs

parser = argparse.ArgumentParser(description='rouge.py')
parser.add_argument('-input', required=True,
                    help="Path to the input file")
parser.add_argument('-mode', required=True,
                    help="Select one of [train, encode, decode] modes")
parser.add_argument('-model_prefix', default='testModel',
                    help="model name to be saved")
parser.add_argument('-model_type', default='bpe',
                    help="unigram, bpe, char, or word")
parser.add_argument('-vocab_size', default='32000',
                    help="size of the vocabulary")
parser.add_argument('-model', default='testModel',
                    help="Path to the model file")
parser.add_argument('-output', default='testModel',
                    help="Path to the reference file")
opt = parser.parse_args()

def train(opt):
    spm.SentencePieceTrainer.Train('--input=' +  opt.input + \
        ' --model_prefix=' + opt.model_prefix +  ' --vocab_size=' + \
        opt.vocab_size + ' --character_coverage=1.0 --model_type=' + \
        opt.model_type)

def model_encode(opt):
    sp = spm.SentencePieceProcessor()
    sp.Load(opt.model)
    count = 0

    outF = codecs.open(opt.output, "w", "utf-8")
    with codecs.open(opt.input, "r", "utf-8") as inp:
        data = inp.read()
    data = data.split("\n")[:-1]
    for datum in data:
        enc = sp.EncodeAsPieces(datum.lower())
        outF.write(' '.join(enc) + "\n")
    outF.close()

def model_decode(opt):
    sp = spm.SentencePieceProcessor()
    sp.Load(opt.model)

    with codecs.open(opt.input, "r", "utf-8") as inp:
        data = inp.read()
    data = data.split("\n")[:-1]

    outF = codecs.open(opt.output, "w", "utf-8")
    for datum in data:
        decoded = sp.DecodePieces(datum.split(' '))
        outF.write(decoded + "\n")
    outF.close()
    

if __name__ == '__main__':
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'encode':
        model_encode(opt)
    elif opt.mode == 'decode':
        model_decode(opt)
    else:
        print("ERROR: mode not recognized")
