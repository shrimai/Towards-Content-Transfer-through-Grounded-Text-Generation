from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, cxtData, tgtData, batchSize, cuda, volatile=False):
        self.src = srcData
        if cxtData:
            self.cxt = cxtData
            assert(len(self.src) == len(self.cxt))
        else:
            self.cxt = None
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        if self.cxt:
            cxtBatch = self._batchify(
                self.cxt[index*self.batchSize:(index+1)*self.batchSize],
                align_right=False)
        else:
            cxtBatch = None
        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None and cxtBatch is None:
            batch = zip(indices, srcBatch)
            batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
            indices, srcBatch = zip(*batch)
        elif tgtBatch is None:
            batch = zip(indices, srcBatch, cxtBatch)
            batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
            indices, srcBatch, cxtBatch = zip(*batch)
        else:
            batch = zip(indices, srcBatch, cxtBatch, tgtBatch)
            batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
            indices, srcBatch, cxtBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(srcBatch), lengths), wrap(cxtBatch), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.src, self.cxt, self.tgt))
        self.src, self.cxt, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
