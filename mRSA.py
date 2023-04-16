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

    :param alpha: The parameter that determines the intensity of the modesty.
    :param speaker_optimality: The parameter that determines the optimality/rationality of the speaker.
    :param priors: The prior probabilities of the states.
    :param plotting: Whether to plot the distributions at each level of the model.
    
    This implementation models the three basic levels of the RSA model: the literal listener, the pragmatic speaker, and the pragmatic listener and for each level you can choose
    whether to model the whole distribution or the distribution for specific parameters (such as utterance, level of expertise, etc.)
    """
    def __init__(self, alpha: float=alpha, speaker_optimality: float=speaker_optimality, priors: list[float]=priors, plotting: bool=False) -> None:
        self.alpha = alpha
        self.speaker_optimality = speaker_optimality
        self.priors = priors
        self.epsilon = 0.0001
        self.utterances = ["terrible", "bad", "good", "amazing"]
        self.levels_of_expertise = ["beginner", "intermediate", "expert"] # also referred to as states

        # this would have to be determined from an experiment
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
        Compute the probability of utterances given the speaker's level of expertise.
        :return: A 3x4 matrix where each row represents the probability of utterances given the speaker's level of expertise.
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
        :param utterance: The utterance to return the distribution over states for.
        :return: A 4x3 matrix where each row represents the distribution over states for the specified utterance or a 1x3 matrix if an utterance is specified.

        >>> model = mRSA(alpha=1.25, speaker_optimality=10)
        >>> model.L0("bad")
        [0.36774194 0.38709677 0.24516129]

        If utterance is not specified, returns the entire distribution over states.
        >>> model.L0()
        [[0.83414634 0.07317073 0.09268293]
         [0.36774194 0.38709677 0.24516129]
         [0.15019763 0.4743083  0.37549407]
         [0.0785124  0.37190083 0.54958678]]

        """

        # compute P(u|s) -- literal meaning
        P_u_given_s = self.P_u_given_s()

        # P(s) -- reshape to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * len(self.utterances)).T
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
        :param level_of_expertise: The level of expertise to return the distribution over messages for.
        :param modesty: The level of modesty of the speaker. Can be a value between 0 and 1 or a value for modesty between 0 and 1.
        :return: A 4x3 matrix where each row represents the distribution over messages for the specified level of expertise or a 1x3 matrix if a level of expertise is specified.

        >>> model = mRSA(alpha=1.25, speaker_optimality=10)
        >>> model.S1("intermediate")
        [0.10120833 0.76438927 0.12681512 0.00758729]

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
        modest_utility = np.array([list(- sum(self.alpha * state * state_probability for state, state_probability in enumerate(L0[i], start=1)) for i, _ in enumerate(L0))] * len(self.levels_of_expertise))
        
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
        or to infer the modesty value given the level of expertise. Both can be left unspecified to infer both the level of expertise and the modesty value.
        
        :param utterance: The utterance to infer the level of expertise or the modesty value for.
        :param true_state: The true state of the speaker. Can be "good", "intermediate", or "bad".
        :param known_modesty: The level of modesty of the speaker. Can be a value between 0 and 1.

        If both true_state and known_modesty are specified, returns the probability of the true state given the known modesty.
        """
        # infer the probability of the true state given the known modesty
        if true_state is None and known_modesty is None:
            return self._L1_infer_state_and_modesty(utterance=utterance)

        # infer the level of expertise and the modesty value
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

        :param utterance: The utterance to infer the probability of the level of expertise for given the modesty value.
        :param true_state: The true state of the speaker. Can be "good", "intermediate", or "bad".
        :param known_modesty: The level of modesty of the speaker. Can be a value between 0 and 1.
        :return: The probability of the true state given the known modesty.
        """
        if self.plotting:
            print("Nothing to plot, as it's only one probability value.")
            self.plotting = False
            S1 = self.S1(modesty=known_modesty)
            self.plotting = True
        else:
            S1 = self.S1(modesty=known_modesty)
        return S1[self.levels_of_expertise.index(true_state)][self.utterances.index(utterance)]

    
    def _L1_infer_state(self, utterance: str, known_modesty: float) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over true states given an utterance and known goal weights.

        :param utterance: The utterance to infer the level of expertise for given the modesty value.
        :param known_modesty: The level of modesty of the speaker. Can be a value between 0 and 1.
        :return: A 1x3 matrix where each row represents the distribution over states for the specified utterance.
        """
        if self.plotting:
            self.plotting = get_plot_everything()
            S1 = self.S1(modesty=known_modesty)
            self.plotting = True
        else:
            S1 = self.S1(modesty=known_modesty)

        # tranforming priors to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * len(self.utterances)).T
        unnormalized_ouput = S1 * priors
        unnormalized_ouput = unnormalized_ouput.T
        normalized_ouput = normalize(unnormalized_ouput)
        return normalized_ouput[self.utterances.index(utterance)]
    


    def _L1_infer_modesty(self, utterance: str, true_state: str) -> np.ndarray:
        """
        Helper function for L1. Computes the distribution over known goal weights given an utterance and true state.

        :param utterance: The utterance to infer the modesty value for given the level of expertise.
        :param true_state: The true state of the speaker. Can be "good", "intermediate", or "bad".
        :return: A 1x10 matrix where each row represents the distribution over modesty weights for the specified utterance.
        """
        self.plotting = False
        # tranforming priors to correct dimensions for element-wise multiplication
        priors = np.array([self.priors] * len(self.utterances)).T

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

        :param utterance: The utterance to infer the level of expertise and the modesty value for.
        :return: A 3x10 matrix where each row represents a level of expertise and the columns represent specific modesty values.
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