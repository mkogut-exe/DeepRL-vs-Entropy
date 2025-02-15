from Wordle import Environment, check_letters_maches
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import os

def replace_value(array, old_value, new_value):
    array[array == old_value] = new_value
    return array


class Entropy_maximizer:
    def __init__(self, env:Environment, silent=False):
        self.env = env
        self.silent = silent


        #print(env.word)


    def maximizer_guess(self, guess, debug=False):
        matches = self.env.guess(guess)


        if debug:
            print(self.env.guesses)
            print(self.env.guess_maches)
            print("maximizer_guess")

    def play_max_entropy(self,target_word=None, debug=False):

        if target_word is None:
            self.env.reset()
        else:
            self.env.reset(target_word)
        if not self.silent:
            print(self.env.word)
        matches = self.env.allowed_words
        while self.env.try_count < self.env.max_tries:

            entropies = None

            if self.env.win:
                if not self.silent:
                    print('Win!')

                break
            else:
                if not self.silent:
                    print(f'\nTurn: {self.env.try_count + 1}')

            if self.env.try_count == 0:
                if os.path.exists(f'{self.env.path_name}_Entropy_list.pkl'):
                    if not self.silent:
                        print(f"File {self.env.path_name}_Entropy_list.pkl exists. "
                          f"Importing entropy list.")
                    sorted_Entropy_list = self.load_starting_entropy_list()
                else:
                    if not self.silent:
                        print(f"File {self.env.path_name}_Entropy_list.pkl does not exist. "
                          f"Creating entropy list and saving to the file.")
                    sorted_Entropy_list = self.save_starting_entropy_list()

                self.maximizer_guess(sorted_Entropy_list[0][1])
                if not self.silent:
                    print(f'    Guess: {sorted_Entropy_list[0][1]}, Entropy: {sorted_Entropy_list[0][0]}')
            else:
                if self.env.try_count >= 1:
                    matches=self.better_find_matches(word_list=matches)
                    guess, entropy, entropies = self.calculate_entropy(matches)

                    if debug:
                        sorted_entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
                        if not self.silent:
                            for i, (entropy, word) in enumerate(sorted_entropies):
                                print(f'{i + 1}. {word}: {entropy}')
                    self.maximizer_guess(guess)
                    if not self.silent:
                        print(f'    Guess: {guess}, Entropy: {entropy}')

        if not self.env.win:
            if not self.silent:
                print('\nLoss!')
                print(f'The word was: {self.env.word}')
                print(f'Matching words with entropies at the end of te game:\n')
                sorted_entropies=sorted(entropies, key=lambda x: x[0],reverse=True)
                for i, (entropy, word) in enumerate(sorted_entropies):
                    print(f'{i + 1}. {word}: {entropy}')
        return self.env.win, self.env.try_count



    def better_find_matches(self,word_list=None, debug=False):
        guesses = self.env.guesses.tolist()

        if word_list is None:
            word_list = self.env.allowed_words
        total_matching = []
        if len(guesses) == 0:
            return word_list.tolist()

        zero_letters=set()
        present_letters=set()
        absent_letters=set()

        alphabet_array_matches_final = np.zeros(26, dtype=int)

        for t in range(len(self.env.guess_maches)):
            alphabet_array_matches = np.zeros(26, dtype=int)

            for i in range(self.env.word_length):
                if self.env.guess_maches[t][i] >= 1:
                    alphabet_array_matches[ord(guesses[t][i]) - ord('a')] += 1
                    present_letters.add(guesses[t][i])

                elif self.env.guess_maches[t][i] == 0 and guesses[t][i] not in present_letters:
                    zero_letters.add(guesses[t][i])

            for i in range(len(alphabet_array_matches)):
                alphabet_array_matches_final[i] = max(alphabet_array_matches_final[i], alphabet_array_matches[i])

        if debug:
            print(f'zero_letters: {zero_letters}')
            print(f'present_letters: {present_letters}')

        absent_letters=zero_letters - present_letters
        if not self.silent:
            print(f' absent_letters: {absent_letters}')
        for candidate in word_list:
            #print(candidate,absent_letters.isdisjoint(candidate))
            if absent_letters.isdisjoint(candidate):
                alphabet_array_candidate = np.zeros(26, dtype=int)

                candidate_feedback = check_letters_maches(guesses[-1], candidate)

                candidate_correct_position_matches = replace_value(candidate_feedback.copy(), 1, 0)
                maximizer_correct_position_matches = replace_value(self.env.guess_maches[-1].copy(), 1, 0)



                if np.array_equal(candidate_correct_position_matches,maximizer_correct_position_matches):

                    matching_letters = False

                    """if candidate == 'tizes':
                       print(candidate_feedback, candidate)
                       print(guesses[-1])"""

                    for i in range(self.env.word_length):

                        if candidate_feedback[i] > 0:
                            alphabet_array_candidate[ord(guesses[-1][i]) - ord('a')] += 1


                    #print(candidate, alphabet_array_candidate, alphabet_array_matches_final)
                    """if candidate == 'tizes':
                        print(candidate, alphabet_array_candidate, alphabet_array_matches_final)
                        print()"""

                    for i in range(len(alphabet_array_candidate)):

                        if alphabet_array_candidate[i] >= alphabet_array_matches_final[i]:
                            matching_letters = True
                        else:

                            break

                    if matching_letters:
                        total_matching.append(candidate)

        return total_matching

    def calculate_entropy(self, words=None):
        Entropy_list = []
        if words is None:
            words= self.env.allowed_words.tolist()

        max_entropy = -np.inf
        best_guess = None
        word_total = len(words)

        for i in tqdm(range(len(words)), disable=self.silent):
            candidate = words[i]
            pattern_counts = defaultdict(int)

            for word in words:
                match = check_letters_maches(candidate, word)
                feedback_tuple = tuple(match)

                pattern_counts[feedback_tuple] += 1

            #print(pattern_counts)
            entropy = 0.0
            for count in pattern_counts.values():
                p = count / word_total
                if p > 0:
                    entropy += p * np.log2(1 / p)
            Entropy_list.append((entropy, candidate))

            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = candidate

        return best_guess, max_entropy, Entropy_list
    def save_starting_entropy_list(self):
        best_guess, max_entropy, Entropy_list = self.calculate_entropy()
        sorted_Entropy_list = sorted(Entropy_list, key=lambda x: x[0],reverse=True)
        with open(f'{self.env.path_name}_Entropy_list.pkl', 'wb') as f:
            pickle.dump(sorted_Entropy_list, f)
        return sorted_Entropy_list

    def load_starting_entropy_list(self):
        with open(f'{self.env.path_name}_Entropy_list.pkl', 'rb') as f:
            sorted_Entropy_list = pickle.load(f)
        return sorted_Entropy_list
    def print_Entropy_list(self):
        sorted_Entropy_list = self.load_starting_entropy_list()
        print('\n')
        if os.path.exists(f'{self.env.path_name}_Entropy_list.pkl'):
            print(f"File {self.env.path_name}_Entropy_list.pkl exists. Importing entropy list.")
            sorted_Entropy_list = self.load_starting_entropy_list()
        else:
            print(f"File {self.env.path_name}_Entropy_list.pkl does not exist. "
                  f"Creating entropy list and saving to the file.")
            sorted_Entropy_list = self.save_starting_entropy_list()
        print(f'Entropy list:\n')
        for i, (entropy, word) in enumerate(sorted_Entropy_list):
            print(f'{i + 1}. {word}: {entropy}')

    def get_stats(self, save_path=None):
        if save_path and os.path.exists(f'{save_path}.pkl'):
            print(f"Loading stats from {save_path}.pkl")
            with open(f'{save_path}.pkl', 'rb') as f:
                stats=pickle.load(f)
            print(stats)
            return stats
        results = {}
        # Changed to start from 0 to include first-try wins
        tries_distribution = {i: 0 for i in range(0, self.env.max_tries + 2)}
        total_games = 0
        wins = 0

        for word in tqdm(self.env.allowed_words):
            win, tries = self.play_max_entropy(target_word=word)
            results[word] = {"win": win, "tries": tries}
            total_games += 1
            if win:
                wins += 1
                tries_distribution[tries] += 1
            else:
                tries_distribution[self.env.max_tries + 1] += 1

        self.game_stats = {
            "results": results,
            "total_games": total_games,
            "wins": wins,
            "win_rate": wins / total_games * 100,
            "tries_distribution": tries_distribution
        }

        if save_path is not None:
            print(f"Saving stats to {save_path}.pkl")
            with open(f'{save_path}.pkl', 'wb') as f:
                pickle.dump(self.game_stats, f)
        print(self.game_stats)
        return self.game_stats





em=Entropy_maximizer(Environment('wordle-nyt-allowed-guesses-update-12546.txt'),silent=False)

#em.play_max_entropy()

#em.get_stats('wordle_stats')

"""em.env.reset('vuggs')
em.maximizer_guess('abcde')
matches=em.better_find_matches()
print(len(matches),matches)

em.maximizer_guess('efghi')
matches=em.better_find_matches()
print(len(matches),matches)
em.maximizer_guess('jklmn')
matches=em.better_find_matches()
print(len(matches),matches)

em.maximizer_guess('oprst')
matches=em.better_find_matches()
print(len(matches),matches)

em.maximizer_guess('uwxyz')
matches=em.better_find_matches()
print(len(matches),matches)

em.maximizer_guess('bumfs')
matches=em.better_find_matches()
print(len(matches),matches)"""
#em.print_Entropy_list()