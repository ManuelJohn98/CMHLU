"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
import numpy as np
from utils import plotter, normalize, adjust_modesty, get_honesty_from_modesty, get_plot_everything

# Hyperparameters
alpha = 1
speaker_optimality = 1

priors = [1/3, 1/3, 1/3]



class mRSA:
    """
    An implementation of the RSA model extending it to account for modesty.
    """
    def __init__(self, alpha: float=alpha, speaker_optimality: float=speaker_optimality, priors: list[float]=priors, plotting: bool=False) -> None:
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
        self.weight_bins = np.linspace(0.00001, 1, 10, endpoint=True)
        self.plotting = plotting



    def P_u_given_s(self) -> np.ndarray:
        """
        Compute the probability of utterances given speaker's level of expertise.
        """
        # literal truth values
        unnorm = self.literal_semantics

        # normalize to obtain a probability distribution
        norm = normalize(unnorm)

        return norm
    


    @plotter
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
        normalized_output = normalize(unnormalized_output)

        # if no utterance is specified, return the entire distribution
        if utterance is None:
            return normalized_output
        # otherwise, return the distribution for the specified utterance
        return normalized_output[self.utterances.index(utterance)]
    


    @plotter
    def S1(self, level_of_expertise: str|None=None, modesty: float=0.8) -> np.ndarray:
        """
        Implementation of the pragmatic speaker: S1(m|o). Can be given a level of expertise to return the distribution over messages for that level of expertise.
        >>> mRSA().S1("beginner")
        >>> TODO: add expected output

        Can also be a value for modesty between 0 and 1. This determines how modest the speaker is. From  0.5 to 1, the speaker is increasingly modest to the
        point where she will lie and rate her product as lower as would be expected from her level of expertise. From 0.5 to 0, the speaker will start to brag
        about her product to the point where she will rate her product as higher than would be expected from her level of expertise.
        Modesty also determines the level of honesty of the speaker. modesty=0.5 is the most honest, while modesty=0 and modesty=1 are the least honest. 
        The default value for modesty is 0.8.
        """
        honesty = get_honesty_from_modesty(modesty)
        if self.plotting:
            self.plotting = get_plot_everything()
            L0 = self.L0()
            self.plotting = True
        else:
            L0 = self.L0()
        epistemic_utility = np.log(L0.T)

        # modest utility: negative expected state of literal listener
        modest_utility = np.array([list(- sum(self.alpha * state * state_probability for state, state_probability in enumerate(L0[i], start=1)) for i, _ in enumerate(L0))] * 3)
        
        # general utility with adjusted modesty weight
        utility = honesty * epistemic_utility + adjust_modesty(modesty) * modest_utility
        unnormalized_output = np.exp(self.speaker_optimality*utility)
        normalized_output = normalize(unnormalized_output)

        # if no level of expertise is specified, return the entire distribution
        if level_of_expertise is None:
            return normalized_output
        # otherwise, return the distribution for the specified level of expertise
        return normalized_output[self.levels_of_expertise.index(level_of_expertise)]
    

    @plotter
    def L1(self,  utterance: str, true_state:str|None=None, known_modesty: float|None=None) -> np.ndarray | tuple[float, float, float] | float | None:
        """
        Implementation of the pragmatic listener: L1(o|m). Needs to be given an utterance to either infer the level of expertise given the modesty value
        or to infer the modesty value given the level of expertise.
        >>> mRSA().L1("terrible")
        >>> TODO: add expected output

        If both true_state and known_modesty are specified, returns the probability of the true state given the known modesty.
        """
        if true_state is None and known_modesty is None:
            return self._L1_infer_state_and_modesty(utterance=utterance)

        if true_state is not None and known_modesty is not None:
            return self._L1_prob(utterance=utterance, true_state=true_state, known_modesty=known_modesty)
        
        # infer the modesty value given the level of expertise
        if true_state is not None:
            return self._L1_infer_modesty(utterance=utterance, true_state=true_state)
        
        # infer the level of expertise given the modesty value
        if known_modesty is not None:
            return self._L1_infer_state(utterance, known_modesty)
        
    def _L1_prob(self, utterance: str, true_state: str, known_modesty: float) -> float:
        """
        Helper function for L1. Computes the probability of the true state given an utterance and known goal weights.
        """
        if self.plotting:
            self.plotting = False
            S1 = self.S1(modesty=known_modesty)
            self.plotting = True
        else:
            S1 = self.S1(modesty=known_modesty)
        return S1[self.levels_of_expertise.index(true_state)][self.utterances.index(utterance)]

    
    def _L1_infer_state(self, utterance: str, known_modesty: float) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over true states given an utterance and known goal weights.
        """
        if self.plotting:
            self.plotting = get_plot_everything()
            S1 = self.S1(modesty=known_modesty)
            self.plotting = True
        else:
            S1 = self.S1(modesty=known_modesty)

        # tranforming priors to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * 4).T
        unnormalized_ouput = S1 * priors
        unnormalized_ouput = unnormalized_ouput.T
        normalized_ouput = normalize(unnormalized_ouput)
        return normalized_ouput[self.utterances.index(utterance)]
    


    def _L1_infer_modesty(self, utterance: str, true_state: str) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over known goal weights given an utterance and true state.
        """
        self.plotting = False
        # tranforming priors to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * 4).T

        likelihood_grid = np.empty(len(self.weight_bins))


        for i, modesty in enumerate(self.weight_bins):
            S1 = self.S1(modesty=modesty)
            unnormalized_output = S1 * priors
            unnormalized_output = unnormalized_output.T
            normalized_output = normalize(unnormalized_output)

            likelihood_grid[i] = normalized_output[self.utterances.index(utterance)][self.levels_of_expertise.index(true_state)]
        

        self.plotting = True
        return likelihood_grid
    

    def _L1_infer_state_and_modesty(self, utterance: str) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over true states and known goal weights given an utterance.
        """

        likelihood_grid = np.empty((len(self.levels_of_expertise), len(self.weight_bins)))

        if self.plotting:
            self.plotting = False

            for i, level_of_expertise in enumerate(self.levels_of_expertise):
                likelihood_grid[i] = self._L1_infer_modesty(utterance=utterance, true_state=level_of_expertise)
            self.plotting = True
        else:
            for i, level_of_expertise in enumerate(self.levels_of_expertise):
                likelihood_grid[i] = self._L1_infer_modesty(utterance=utterance, true_state=level_of_expertise)

        return likelihood_grid




if __name__ == "__main__":
    from main import main
    main()