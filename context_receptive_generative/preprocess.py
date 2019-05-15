import onmt
import argparse
import torch
import codecs

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_cxt', default=None,
                    help="Path to the training context data")                    
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_cxt', default=None,
                    help="Path to the validation context data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=32000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=32000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-cxt_vocab',
                    help="Path to an existing source vocabulary")                    
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-seq_length_src', type=int, default=50,
                    help="Maximum sequence length of source")
parser.add_argument('-seq_length_cxt', type=int, default=50,
                    help="Maximum sequence length of context")
parser.add_argument('-seq_length_tgt', type=int, default=50,
                    help="Maximum sequence length of target")                    
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def makeData(srcFile, cxtFile, tgtFile, srcDicts, cxtDicts, tgtDicts):
    print('Processing %s & %s ...' % (srcFile, tgtFile))
    with codecs.open(srcFile, "r", "utf-8") as inp:
        srcD = inp.read()
    srcD = srcD.split("\n")[:-1]
    if cxtFile:
        with codecs.open(cxtFile, "r", "utf-8") as inp:
            cxtD = inp.read()
        cxtD = cxtD.split("\n")[:-1]
        assert len(srcD) == len(cxtD)
    else:
        cxtD = []
    with codecs.open(tgtFile, "r", "utf-8") as inp:
        tgtD = inp.read()
    tgtD = tgtD.split("\n")[:-1]
    assert len(srcD) == len(tgtD)

    src, cxt, tgt = [], [], []
    sizes = []
    count, ignored = 0, 0

    for idx in range(len(srcD)):
        srcWords = srcD[idx].split()
        if cxtD != []:
            cxtWords = cxtD[idx].split()
        else:
            cxtWords = 0
        tgtWords = tgtD[idx].split()

        if len(srcWords) <= opt.seq_length_src and len(cxtWords) <= opt.seq_length_cxt and len(tgtWords) <= opt.seq_length_tgt:

            src += [srcDicts.convertToIdx(srcWords,
                            onmt.Constants.UNK_WORD)]
            if cxtWords:
                cxt += [cxtDicts.convertToIdx(cxtWords,
                            onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                            onmt.Constants.UNK_WORD,
                            onmt.Constants.BOS_WORD,
                            onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        if cxt != []:
            cxt = [cxt[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    if cxt != []:
        cxt = [cxt[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length_src))

    return src, cxt, tgt

def makeDataOld(srcFile, cxtFile, tgtFile, srcDicts, cxtDicts, tgtDicts):
    src, cxt, tgt = [], [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, "r", "utf-8")
    if cxtFile:
        cxtF = codecs.open(cxtFile, "r", "utf-8")
    tgtF = codecs.open(tgtFile, "r", "utf-8")

    while True:
        sline = srcF.readline()
        if cxtFile:
            cline = cxtF.readline()
        else:
            cline = None
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and cline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or cline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        if cline:
            cline = cline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or cline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        if cline:
            cxtWords = cline.split()
        else:
            cxtWords = 0
        tgtWords = tline.split()

        if len(srcWords) <= opt.seq_length_src and len(cxtWords) <= opt.seq_length_cxt and len(tgtWords) <= opt.seq_length_tgt:

            src += [srcDicts.convertToIdx(srcWords,
                            onmt.Constants.UNK_WORD)]
            if cline:
                cxt += [cxtDicts.convertToIdx(cxtWords,
                            onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                            onmt.Constants.UNK_WORD,
                            onmt.Constants.BOS_WORD,
                            onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    if cxtFile:
        cxtF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        if cxt != []:
            cxt = [cxt[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    if cxt != []:
        cxt = [cxt[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length_src))

    return src, cxt, tgt

def main():

    dicts = {}
    print('Preparing source vocab ....')
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    print('Preparing target vocab ....')
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)
    if opt.train_cxt:
        dicts['cxt'] = dicts['src']
    else:
        dicts['cxt'] = None

    print('Preparing training ...')
    train = {}
    train['src'], train['cxt'], train['tgt'] = makeData(opt.train_src, opt.train_cxt, opt.train_tgt,
                                          dicts['src'], dicts['cxt'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['cxt'], valid['tgt'] = makeData(opt.valid_src, opt.valid_cxt, opt.valid_tgt,
                                    dicts['src'], dicts['cxt'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.cxt_vocab is None:
        saveVocabulary('context', dicts['cxt'], opt.save_data + '.cxt.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                }
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
