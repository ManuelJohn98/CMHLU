"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
import numpy as np
import matplotlib.pyplot as plt


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
            [0.9,       0.3, 0.2, 0.1], # beginner
            [0.1,       0.4, 0.8, 0.6], # intermediate
            [0.1,       0.2, 0.5, 0.7]  # expert
        ])
        self.weight_bins = np.linspace(0, 1, 50, endpoint=False)

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
        # TODO: !!!honesty is determined by modesty with a reversed bell shaped curve -> curve max = max honesty!!!
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
        unnormalized_output = np.exp(self.speaker_optimality*utility)
        normalized_output = self.normalize(unnormalized_output)

        # if no level of expertise is specified, return the entire distribution
        if level_of_expertise is None:
            return normalized_output
        # otherwise, return the distribution for the specified level of expertise
        return normalized_output[self.levels_of_expertise.index(level_of_expertise)]
    
    def L1(self,  utterance: str, true_state:str|None=None, known_goal_weights: tuple[float, float]|None=None) -> np.ndarray | tuple[float, float, float] | None:
        """
        Implementation of the pragmatic listener: L1(o|m). Can be given an utterance to return the distribution over states for that utterance.
        >>> mRSA().L1("terrible")
        >>> TODO: add expected output
        If utterance is not specified, returns the entire distribution over states.
        """
        if true_state is None and known_goal_weights is None:
            raise ValueError("Either true_state or known_goal_weights must be specified.")
        if true_state is not None and known_goal_weights is not None:
            # TODO: calculate probability of true state given known goal weights?
            raise NotImplementedError
        if true_state is not None:
            return self._L1_infer_goal_weights(utterance=utterance, true_state=true_state)
        if known_goal_weights is not None:
            return self._L1_infer_state(utterance, known_goal_weights)

    def _L1_infer_state(self, utterance: str, known_goal_weights: tuple[float, float]) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over true states given an utterance and known goal weights.
        """
        honesty, modesty = known_goal_weights
        S1 = self.S1(honesty=honesty, modesty=modesty)
        priors = np.array([self.priors] * 4).T
        unnormalized_ouput = S1 * priors
        unnormalized_ouput = unnormalized_ouput.T
        normalized_ouput = self.normalize(unnormalized_ouput)
        return normalized_ouput[self.utterances.index(utterance)]
    
    def _L1_infer_goal_weights(self, utterance: str, true_state: str) -> tuple[float, float, float]:
        """
        Helper function for L1. Computes the distribution over known goal weights given an utterance and true state.
        """
        priors = np.array([self.priors] * 4).T
        max_likelihood = (-1, -1, float('-inf'))
        for honesty in self.weight_bins:
            for modesty in self.weight_bins:
                '''if honesty + modesty:
                    continue'''
                S1 = self.S1(honesty=honesty, modesty=modesty)
                unnormalized_output = S1 * priors
                unnormalized_output = unnormalized_output.T
                normalized_output = self.normalize(unnormalized_output)
                if np.argmax(normalized_output[self.utterances.index(utterance)]) != self.levels_of_expertise.index(true_state):
                    continue
                max_likelihood = max(max_likelihood, (honesty, modesty, normalized_output[self.utterances.index(utterance)][self.levels_of_expertise.index(true_state)]), key=(lambda x: x[2]))
        return max_likelihood


if __name__ == "__main__":
    model = mRSA(alpha=1.25, speaker_optimality=10)
    print(model.S1(honesty=0.6, modesty=0.9))
