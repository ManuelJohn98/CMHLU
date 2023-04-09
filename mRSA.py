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

def adjust_modesty(x: float) -> float:
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
        self.levels_of_expertise = ["beginner", "intermediate", "expert"] # also referred to as states
        self.literal_semantics = np.array([
            #terrible, bad, good, amazing
            [0.9,       0.8, 0.4, 0.1], # beginner
            [0.1,       0.4, 0.8, 0.6], # intermediate
            [0.1,       0.2, 0.5, 0.7]  # expert
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

        # normalize to obtain a probability distribution
        norm = self.normalize(unnorm)

        return norm
    
    def L0(self, utterance: str|None=None) -> np.ndarray:
        """
        Implementation of the literal listener: L(s|u). Can be given an utterance to return the distribution over states for that utterance.
        >>> mRSA().L0("terrible")
        >>> # TODO: add expected output
        If utterance is not specified, returns the entire distribution over states.
        """
        # compute P(u|s) -- literal meaning
        P_u_given_s = self.P_u_given_s()

        # P(s) -- reshape to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * 4).T
        unnormalized_output = P_u_given_s * priors

        # transform to obtain a distribution over speakers
        unnormalized_output = unnormalized_output.T

        # replace all 0 values with epsilon
        unnormalized_output[unnormalized_output == 0] = self.epsilon
        normalized_output = self.normalize(unnormalized_output)

        # if no utterance is specified, return the entire distribution
        if utterance is None:
            return normalized_output
        # otherwise, return the distribution for the specified utterance
        return normalized_output[self.utterances.index(utterance)]
        
    def S1(self, level_of_expertise: str|None=None, honesty: float=0.6, modesty : float=0.8) -> np.ndarray:
        """
        Implementation of the pragmatic speaker: S1(m|o). Can be given a level of expertise to return the distribution over messages for that level of expertise.
        >>> mRSA().S1("beginner")
        >>> TODO: add expected output
        Can also be given a either a value for honesty or modesty or both.
        >>> mRSA().S1(honesty=0.6, modesty=0.8)
        >>> TODO: add expected output
        All arguments are optional. If no arguments are specified, returns the entire distribution over messages with default values for honesty and modesty:
        >>> mRSA().S1(honesty=0.6, modesty=0.8)
        """
        L0 = self.L0()
        epistemic_utility = np.log(L0.T)

        # modest utility: negative expected state of literal listener
        modest_utility = np.array([list(- sum(self.alpha * state * state_probability for state, state_probability in enumerate(L0[i], start=1)) for i, _ in enumerate(L0))] * 3)
        
        # general utility with adjusted modesty weight
        utility = honesty * epistemic_utility + adjust_modesty(modesty) * modest_utility
        unnorm = np.exp(self.speaker_optimality*utility)
        norm = self.normalize(unnorm)

        # if no level of expertise is specified, return the entire distribution
        if level_of_expertise is None:
            return norm
        # otherwise, return the distribution for the specified level of expertise
        return norm[self.levels_of_expertise.index(level_of_expertise)]


if __name__ == "__main__":
    model = mRSA(alpha=1.3)
    print(model.L0("terrible"))
