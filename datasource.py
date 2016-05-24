import numpy
from collections import defaultdict
import random

class Datasource():
    """
    Wrapper around a dataset which permits

    1) Iteration over minibatches using next(); call reset() between epochs to randomly shuffle the data
    2) Access to the entire dataset using all()
    """

    def __init__(self, data, worddict, batch_size=128, max_cap_lengh=50):
        self.data = data
        self.batch_size = batch_size
        self.worddict = worddict
        self.num_images = len(self.data['ims'])
        self.parents = defaultdict(set)
        self.max_cap_lengh = max_cap_lengh
        self.reset()


    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(self.num_images)

    def next(self):
        image_ids = []
        caption_ids = []

        while len(image_ids) < self.batch_size:
            image_id = self.order[self.idx]
            caption_id = image_id * 5 + random.randrange(5)
            image_ids.append(image_id)
            caption_ids.append(caption_id)

            self.idx += 1
            if self.idx >= self.num_images:
                self.reset()
                raise StopIteration()

        x = self.prepare_caps(caption_ids)
        im = self.data['ims'][numpy.array(image_ids)]

        return x, im

    def all(self):

        return self.prepare_caps(range(0,len(self.data['caps']))) , self.data['ims']

    def __iter__(self):
        return self

    def prepare_caps(self, indices):
        seqs = []
        for i in indices:
            cc = self.data['caps'][i]
            real = [self.worddict[w] if w in self.worddict else 1 for w in cc.split()]
            padding = [0]*(self.max_cap_lengh-len(real))
            seqs.append(real + padding)

        return numpy.array(seqs)


