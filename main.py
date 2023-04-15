"""Copyright (c) 2023, Manuel John
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. """
from mRSA import mRSA

def main():
    try:
        while True:
            model: mRSA
            print("-----------Hyperparameters-----------")
            try:
                alpha = float(input("Give a value for alpha (default is 1), which regulates the intensity of the modesty value: "))
            except ValueError:
                alpha = 1
            try:
                speaker_optimality = float(input("Give a value for the speaker's optimality (default is 1), which is the rationality parameter: "))
            except ValueError:
                speaker_optimality = 1
            toggle_plotting = input("Do you want to enable plotting? (y/n) (default is n): ")
            if toggle_plotting == "y":
                model = mRSA(alpha=alpha, speaker_optimality=speaker_optimality, plotting=True)
            else:
                model = mRSA(alpha=alpha, speaker_optimality=speaker_optimality)
            while True:
                print("-----------Model-----------")
                level = input("Which level of the model do you want to run? [L0, S1, L1]: ")
                match level:
                    case "L0":
                        print("-----------L0-----------")
                        print("You can specify an utterance {} or just press [ENTER] to get the whole distribution.".format(model.utterances))
                        utterance = input("Utterance: ")
                        if utterance == "":
                            print(model.L0())
                        else:
                            print(model.L0(utterance))
                    case "S1":
                        print("-----------S1-----------")
                        try:
                            modesty = float(input("Give a value for the speaker's modesty (default is 0.8): "))
                        except ValueError:
                            modesty = 0.8
                        print("You can specify a level of expertise {} or just press [ENTER] to get the whole distribution.".format(model.levels_of_expertise))
                        utterance = input("Utterance: ")
                        if utterance == "":
                            print(model.S1(modesty=modesty))
                        else:
                            print(model.S1(utterance, modesty=modesty))
                    case "L1":
                        print("-----------L1-----------")
                        print("You can specify the true level of expertise or the true modesty or both or just press [ENTER] to infer both.")
                        true_level_of_expertise = input("True level of expertise: ")
                        if true_level_of_expertise == "":
                            true_level_of_expertise = None
                        true_modesity = input("True modesty: ")
                        if true_modesity == "":
                            true_modesity = None
                        utterance = input("You need to specify an utterance from {}: ".format(model.utterances))
                        print(model.L1(utterance, true_level_of_expertise, true_modesity))
                    case _:
                        print("Invalid input.")
                        continue
                try_again = input("Do you want to try another \u0332l\u0332evel, chose new \u0332h\u0332yperparameters or e\u0332x\u0332it? (l/h/x): ").lower()
                if try_again == "l":
                    continue
                elif try_again == "h":
                    break
                elif try_again == "x":
                    exit()
    except KeyboardInterrupt:
        print("")
        print("Exiting...")
        exit()