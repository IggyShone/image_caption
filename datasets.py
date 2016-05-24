"""
Dataset loading
"""
import numpy
from collections import OrderedDict


def load_dataset(path, cnn, fold=0):
    """
    Load captions and image features
    Possible options: coco
    """
    splits = ['train', 'test', 'dev']

    dataset = {}

    for split in splits:
        dataset[split] = {}
        caps = []
        splitName = 'val' if split == 'dev' else split
        with open('%s/%s.txt' % (path, splitName), 'rb') as f:
            for line in f:
                caps.append(line.strip())
            dataset[split]['caps'] = caps

        dataset[split]['ims'] = numpy.load('%s/images/%s/%s.npy' % (path, cnn, splitName))

        # handle coco specially by only taking 1k or 5k captions/images
        if split in ['dev', 'test']:
            dataset[split]['ims'] = dataset[split]['ims'][fold*1000:(fold+1)*1000]
            dataset[split]['caps'] = dataset[split]['caps'][fold*5000:(fold+1)*5000]

    return dataset


def build_dictionary(text):

    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2  # 0: <eos>, 1: <unk>

    return worddict

