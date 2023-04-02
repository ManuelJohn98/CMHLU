"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
import numpy as np
from functools import partial

# hyper parameters
# alpha = 1.25
lamb = 10

states = range(1,6)

utterances = ["terrible", "bad", "okay", "good", "amazing"]
index = partial(utterances.index)

literal_semantics = np.array(
            [[.95,.85,.02,.02,.02],
            [.85,.95,.02,.02,.02],
            [0.02,0.25,0.95,.65,.35],
            [.02,.05,.55,.95,.93],
            [.02,.02,.02,.65,0.95]]
        )

class mRSA:
    def __init__(self, lamb=10):
        self.lamb = lamb


    def meaning(self, utterance: str, state: int) -> bool:
        return bool(np.random.binomial(1, literal_semantics[index(utterance)][state - 1])) if utterance in utterances else True


    def normalize(self,arr):
	    return arr / arr.sum(axis=1)[:, np.newaxis]


    def P_m_given_o(self):
        unnorm = literal_semantics

        #normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm


    def L0(self, utterance: str) -> int:
        pass




if __name__=='__main__':
    pass
