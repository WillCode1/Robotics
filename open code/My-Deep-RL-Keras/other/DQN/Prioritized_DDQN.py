import heapq
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym


class PrioritizedExperienceQueue:
    """ Prioritized replay memory using binary heap """

    def __init__(self, maxlen=10000):
        self.maxlen = maxlen
        self.memory = []

    def size(self):
        return len(self.memory)

    def append(self, experience, TDerror):
        heapq.heappush(self.memory, (experience, -TDerror))
        if self.size() > self.maxlen:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def batch(self, batch_size):
        batch = heapq.nsmallest(batch_size, self.memory)
        batch = [e for (_, e) in batch]
        self.memory = self.memory[batch_size:]
        return batch
