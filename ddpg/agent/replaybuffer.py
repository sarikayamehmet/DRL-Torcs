
from collections import namedtuple
from random import sample as randsample
import logging

Transition = namedtuple("Transition", ("state",
                                        "action",
                                        "reward",
                                        "next_state",
                                        "terminal")
                                    )


class UniformBuffer(object):

    def __init__(self, size, logger_level=logging.WARNING):
        self.queue = []
        self.logger = logging.getLogger(__name__ + __class__.__name__)
        self.logger.setLevel(logger_level)
        self.cycle = 0
        self.size = size

    def __len__(self):
        return len(self.queue)

    def push(self, **transition):
        if self.size != len(self.queue):
            self.queue.append(Transition(**transition))
        else:
            self.queue.append(Transition(**transition))
            self.cycle  = (self.cycle + 1)%self.size
    
    def sample(self, batchsize):
        if batchsize > len(self.queue):
            self.logger.debug("Batchsize is to big for queue size: {}".format(len(self.queue)))
            return None
        batch = randsample(self.queue, batchsize)
        return Transition(*zip(*batch))