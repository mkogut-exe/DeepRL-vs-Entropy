from Wordle import Environment, check_letters_maches
import numpy as np

def replace_value(array, old_value, new_value):
    array[array == old_value] = new_value
    return array


class Entropy_maximizer:
    def __init__(self, env:Environment):
        self.env = env

        print(env.word)

    def maximizer_guess(self, guess):
        self.env.guess(guess)
        print(self.env.guesses)
        print(self.env.guess_maches)
        print("maximizer_guess")
    def probability_calculation(self):
        total_matching = 0


        for candidate in self.env.allowed_words:
            matching = False
            alphabet_array_candidate = np.zeros(26, dtype=int)
            alphabet_array_matches_final = np.zeros(26, dtype=int)

            candidate_feedback = check_letters_maches(self.env.guesses[-1], candidate)


            candidate_correct_position_matches=replace_value(candidate_feedback.copy(), 1, 0)
            maximizer_correct_position_matches=replace_value(self.env.guess_maches[-1].copy(), 1, 0)

            if np.array_equal(candidate_correct_position_matches,maximizer_correct_position_matches):
                matching = True
            else:
                matching = False

            for i in range(self.env.word_length):

                if candidate_feedback[i]==1:
                    alphabet_array_candidate[ord(self.env.guesses[-1][i]) - ord('a')] += 1


            for t in range(len(self.env.guess_maches)):
                alphabet_array_matches = np.zeros(26, dtype=int)
                for i in range(self.env.word_length):
                    if self.env.guess_maches[t][i] == 1:
                        alphabet_array_matches[ord(self.env.guesses[t][i]) - ord('a')] += 1

                for i in range(len(alphabet_array_matches)):
                    print(alphabet_array_matches_final[i])
                    print(alphabet_array_matches[i])

                    alphabet_array_matches_final[i]=max(alphabet_array_matches_final[i], alphabet_array_matches[i])
                    print('pppp', i)
                print(alphabet_array_matches)
                print(alphabet_array_matches_final)

            if np.array_equal(alphabet_array_matches_final, alphabet_array_candidate):
                matching=True
            else:
                matching = False

            if matching:
                total_matching += 1

            break

        return None

em=Entropy_maximizer(Environment('wordle-nyt-allowed-guesses-update-12546.txt'))
em.maximizer_guess('brail')
em.maximizer_guess('braai')
em.probability_calculation()