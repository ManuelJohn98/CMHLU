"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from functools import partial



def adjust_modesty(x: float) -> float:
    """
    A helper function that linearly maps the modesty value between 0 and 1 to a value between -1 and 1.

    :param x: The modesty value between 0 and 1
    :return: The adjusted modesty value between -1 and 1

    >>> adjust_modesty(0.5)
    0
    >>> adjust_modesty(1)
    1
    >>> adjust_modesty(0)
    -1
    """
    return 2 * (x - 0.5)



def get_honesty_from_modesty(x: float) -> float:
    """
    A helper function that maps the modesty value between 0 and 1 to a value between 0 and 1, where input 1 and 0 return 0 and input 0.5 returns 1.

    :param x: The modesty value between 0 and 1
    :return: The honesty value between 0 and 1

    >>> get_honesty_from_modesty(0.5)
    1
    >>> get_honesty_from_modesty(1)
    0
    """
    if x < 0.5:
        return 2 * (x - 0.25) + 0.5
    elif x > 0.5:
        return -2 * (x - 0.75) + 0.5
    return 1



def get_plot_everything() -> bool:
    """
    A helper function that retrieves from the user whether the model should plot all lower levels of the model.
    """
    while True:
        user_input = input("Do you want to plot all lower levels of the model in succession? (You need to close each window before the next one can appear) (y/n): ")
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        


def normalize(arr: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array by row (i.e. each row sums to 1).
        :param arr: The array to normalize
        :return: The normalized array
        """
        return arr / arr.sum(axis=1)[:, np.newaxis]
    


def plotter(func: Callable) -> Callable:
    """
    A helper function for plotting.
    :param func: A function representing a level of the mRSA model.
    :return: A wrapper function that plots the result of the function if the model is set to plotting.
    """
    def wrapper(model, *args, **kwargs) -> np.ndarray:
        result = func(model, *args, **kwargs)
        if not model.plotting:
            return result
        # level 1 speaker - complete distribution
        if func.__name__=='S1' and len(args) == 0 and (len(kwargs) == 0 or len(kwargs) == 1 and "modesty" in kwargs):
            modesty = kwargs["modesty"] if len(kwargs) == 1 else 0.8
            honesty = get_honesty_from_modesty(modesty)
            X = np.arange(result.shape[1])
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1) # type: ignore
            for i in range(len(result)):
                ax.bar(X + 1/len(result) * i, result[i], width=1/len(result))
            ax.set_title("Distribution of utterances per level of expertise\nwith mod=" + str(round(modesty, 2)) + " and hon=" + str(round(honesty, 2)))
            ax.set_xlabel("Levels of expertise")
            ax.set_xticks(np.arange(result.shape[1]) + 0.25)
            ax.set_xticklabels(model.utterances)
            ax.set_ylabel("Probability inferred by " + r"$S_1$")
            ax.legend(model.levels_of_expertise)
            ax.grid(axis='y', zorder=0)
            plt.savefig("S1_complete.png")
            plt.show()
        # literal listener - complete distribution
        elif func.__name__=='L0' and len(args) == 0 and len(kwargs) == 0:
            X = np.arange(result.shape[1])
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1) # type: ignore
            for i in range(len(result)):
                ax.bar(X + 1/len(result) * i, result[i], width=1/len(result))
            ax.set_title("Distribution of levels of expertise per utterance")
            ax.set_xlabel("Utterances")
            ax.set_xticks(np.arange(result.shape[1]) + 0.5)
            ax.set_xticklabels(model.levels_of_expertise)
            ax.set_ylabel("Probability inferred by " + r"$L_0$")
            ax.legend(model.utterances)
            ax.grid(axis='y', zorder=0)
            plt.savefig("L0_complete.png")
            plt.show()
        # literal listener - single utterance
        elif func.__name__=='L0':
            utterance: str
            try:
                utterance = args[0]
            except IndexError:
                utterance = kwargs["utterance"]
            plt.title(r'$L_0 (s |$' + '"' + utterance + '"' + r'$) \propto P(u | s) * P(s)$')
            plt.bar(model.levels_of_expertise, result)
            plt.xlabel("Levels of expertise")
            plt.ylabel("Probability inferred by " + r"$L_0$")
            plt.grid(axis='y', zorder=0)
            plt.savefig("L0_" + utterance + ".png")
            plt.show()
        # level 1 speaker - single utterance
        elif func.__name__=='S1':
            level_of_expertise: str
            try:
                level_of_expertise = args[0]
            except IndexError:
                level_of_expertise = kwargs["level_of_expertise"]
            modesty: float
            try:
                modesty = args[1]
            except IndexError:
                modesty = kwargs["modesty"] if "modesty" in kwargs else 0.8
            plt.title(r'$S_1 (u | $' + '"' + level_of_expertise + '"' + r'$) \propto exp(speaker-optimality * U(s;u;hon=$' + str(round(get_honesty_from_modesty(modesty), 2)) + r'$;mod=$' + str(round(modesty, 2)) + '$))$')
            plt.xlabel("Utterances")
            plt.ylabel("Probability inferred by " + r"$S_1$")
            plt.bar(model.utterances, result)
            plt.grid(axis='y', zorder=0)
            plt.savefig("S1_" + level_of_expertise + ".png")
            plt.show()
            # pragmatic listener
        elif func.__name__=='L1':
            # infer modesty
            if len(args) == 2 and args[1] in model.levels_of_expertise or len(kwargs) == 1 and "true_state" in kwargs:
                utterance = args[0]
                true_state: str
                try:
                    true_state = args[1]
                except IndexError:
                    true_state = kwargs["true_state"]
                plt.title(r'$L_1 (mod| $' + true_state + '$;$' + '"' + utterance + '"' + r'$) \propto S_1(u | s) * P(s) * P(mod)$')
                plt.xlabel("Weights for modesty")
                plt.ylabel("Probability inferred by " + r"$L_1$")
                plt.bar(model.weight_bins, result, width=1/len(result))
                plt.grid(axis='y', zorder=0)
                plt.savefig("L1_" + true_state + "_" + utterance + ".png")
                plt.show()
            # infer state
            elif len(args) == 2 or len(kwargs) == 1 and "known_modesty" in kwargs:
                utterance = args[0]
                known_modesty: float
                try:
                    known_modesty = args[1]
                except IndexError:
                    known_modesty = kwargs["known_modesty"]
                plt.title(r'$L_1 (s| $' + '"' + utterance + '"' + r'$;$' + str(round(known_modesty, 2)) + r'$) \propto S_1(u|s;mod=$' + str(round(known_modesty, 2)) + r'$) * P(mod) * P(s) $')
                plt.xlabel("Levels of expertise")
                plt.ylabel("Probability inferred by " + r"$L_1$")
                plt.bar(model.levels_of_expertise, result)
                plt.grid(axis='y', zorder=0)
                plt.savefig("L1_" + utterance + "_" + str(round(known_modesty, 2)) + ".png")
                plt.show()
            # infer both
            elif len(args) == 1 and len(kwargs) == 0:
                utterance = args[0]
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1) # type: ignore
                X = np.arange(result.shape[1])
                for i in range(len(result)):
                    ax.bar(X + 1/len(result) * i, result[i], width=1/len(result), align='center')
                ax.set_title(r'$L_1 (s, mod| $' + '"' + utterance + '"' + r'$) \propto S_1(u|s;mod) * P(mod) * P(s) $')
                ax.set_xlabel("Levels of modesty")
                ax.set_xticks(np.arange(result.shape[1]) + 0.33)
                ax.set_xticklabels(map(partial(round, ndigits=2), model.weight_bins))
                ax.set_ylabel("Probability inferred by " + r"$L_1$")
                ax.legend(model.levels_of_expertise)
                ax.grid(axis='y', zorder=0)
                plt.savefig("L1_" + utterance + "_complete.png")
                plt.show()
        return result
    return wrapper


if __name__=='__main__':
    from mRSA import mRSA
    model = mRSA(alpha=1.25, speaker_optimality=10, plotting=True)
    print(model.L0())
    print(model.L0("bad"))
    print(model.S1())
    print(model.S1(modesty=0.6))
    print(model.S1(modesty=0.15))
    print(model.S1("expert", modesty=0.9))
    print(model.S1("intermediate", modesty=0.9))
    print(model.L1("bad"))
    print(model.L1("terrible", known_modesty=0.8))
    print(model.L1("good", true_state="expert"))
    print(model.L1("bad", true_state="expert", known_modesty=0.6))