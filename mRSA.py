"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
import numpy as np
import matplotlib.pyplot as plt
import random
#from typing import List

# Hyperparameters
alpha = 1
speaker_optimality = 1

priors = [1/3, 1/3, 1/3]

def adjusted_modesty(x: float) -> float:
    """
    A function that linearly maps the modesty value between 0 and 1 to a value between -1 and 1.
    """
    return 2 * (x - 0.5)

class mRSA:
    """
    An implementation of the RSA model extending it to account for modesty.
    """
    def __init__(self, alpha: float=alpha, speaker_optimality: float=speaker_optimality, priors: list[float]=priors) -> None:
        self.alpha = alpha
        self.speaker_optimality = speaker_optimality
        self.priors = priors
        self.epsilon = 0.0001
        self.utterances = ["terrible", "bad", "good", "amazing"]
        self.levels_of_expertise = ["beginner", "intermediate", "expert"]
        self.literal_semantics = np.array([
            [0.9, 0.8, 0.4, 0.1], 
            [0.1, 0.4, 0.8, 0.6], 
            [0.1, 0.2, 0.5, 0.7]
        ])

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array to sum to 1.
        """
        return arr / arr.sum(axis=1)[:, np.newaxis]
    
    def P_u_given_s(self) -> np.ndarray:
        """
        Compute the probability of utterances given speaker's level of expertise.
        """
        # literal truth values
        unnorm = self.literal_semantics

        #normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm
    
    def L0(self, utterance: str|None=None) -> np.ndarray:
        """
        Implementation of the literal listener.
        """
        P_u_given_s = self.P_u_given_s()
        priors = np.array([self.priors] * 4).T
        unnorm = P_u_given_s * priors
        unnorm = unnorm.T
        unnorm[unnorm == 0] = self.epsilon
        norm = self.normalize(unnorm)
        if utterance is None:
            return norm
        return norm[self.utterances.index(utterance)]
        
    def S1(self, level_of_expertise: str|None=None, honesty: float=0.6, modesty : float=0.8) -> np.ndarray:
        """
        Implementation of the pragmatic speaker.
        """
        L0 = self.L0()
        epistemic_utility = np.log(L0.T)
        modest_utility = np.array([list(- sum(self.alpha * state * state_probability for state, state_probability in enumerate(L0[i], start=1)) for i, _ in enumerate(L0))] * 3)
        utility = honesty * epistemic_utility + adjusted_modesty(modesty) * modest_utility
        unnorm = np.exp(self.speaker_optimality*utility)
        norm = self.normalize(unnorm)
        if level_of_expertise is None:
            return norm
        return norm[self.levels_of_expertise.index(level_of_expertise)]


if __name__ == "__main__":
    model = mRSA(alpha=1.3)
    print("S1 beginner", model.S1("beginner", honesty=0.3, modesty=0.3))
