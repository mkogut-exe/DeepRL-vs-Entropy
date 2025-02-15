from Wordle import Environment, check_letters_maches
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import os

def replace_value(array, old_value, new_value):#replaces the old value with the new value in the array
    array[array == old_value] = new_value
    return array


class Entropy_maximizer:#class that maximizes the entropy of the guesses and usses it to guess the word
    def __init__(self, env:Environment, silent=False):
        self.env = env #create an instance of the Environment class
        self.silent = silent#variable that determines if the game is played in silent mode (no print statements)


        #print(env.word)


    def maximizer_guess(self, guess, debug=False):#function that makes a guess based on the entropy of the guesses
        matches = self.env.guess(guess)


        if debug:
            print(self.env.guesses)
            print(self.env.guess_maches)
            print("maximizer_guess")

    def play_max_entropy(self,target_word=None, debug=False):#function that plays the game using the max entropy of the guesses

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



    def better_find_matches(self,word_list=None, debug=False):#function that finds the words that match the feedback of the guesses
        guesses = self.env.guesses.tolist()#list of past guesses

        if word_list is None:#if no list of words is provided, the list of allowed words is used
            word_list = self.env.allowed_words
        total_matching = []
        if len(guesses) == 0:
            return word_list.tolist()

        zero_letters=set()#set of letters that have been valued 0 in the feedback of the guesses
        present_letters=set()#set of letters that are in the target word (feedback of the guesses)
        absent_letters=set()#set of letters that are not in the target word (feedback of the guesses)

        alphabet_array_matches_final = np.zeros(26, dtype=int)#array that stores the number of times each letter has been valued 1 in the feedback in all past guesses

        for t in range(len(self.env.guess_maches)):#for each past guess
            alphabet_array_matches = np.zeros(26, dtype=int)#temporary array that stores the number of times each letter has been valued 1 in the feedback of the guess

            for i in range(self.env.word_length):#for each letter in the guess
                if self.env.guess_maches[t][i] >= 1:#if the letter has been valued 1 or 2 in the feedback
                    alphabet_array_matches[ord(guesses[t][i]) - ord('a')] += 1
                    present_letters.add(guesses[t][i])#add the letter to the set of present letters

                elif self.env.guess_maches[t][i] == 0 and guesses[t][i] not in present_letters:#if the letter has been valued 0 in the feedback and is not in the set of present letters
                    zero_letters.add(guesses[t][i])#add the letter to the set of zero letters

            for i in range(len(alphabet_array_matches)):
                alphabet_array_matches_final[i] = max(alphabet_array_matches_final[i], alphabet_array_matches[i])#update the array that stores the number of times each letter has been valued 1 in the feedback in all past guesses

        if debug:
            print(f'zero_letters: {zero_letters}')
            print(f'present_letters: {present_letters}')

        absent_letters=zero_letters - present_letters #set of letters that are not in the target word (feedback of the guesses)
        if not self.silent:
            print(f' absent_letters: {absent_letters}')
        for candidate in word_list:#for each word in the list of suspected matching words
            if absent_letters.isdisjoint(candidate):#checks if the word has any of the letters that are not in the target word and skips the word if it has
                alphabet_array_candidate = np.zeros(26, dtype=int)

                candidate_feedback = check_letters_maches(guesses[-1], candidate)

                #get the guaranteed positions feedback of the guess and candidates
                candidate_correct_position_matches = replace_value(candidate_feedback.copy(), 1, 0)
                maximizer_correct_position_matches = replace_value(self.env.guess_maches[-1].copy(), 1, 0)



                if np.array_equal(candidate_correct_position_matches,maximizer_correct_position_matches):#checks if position of the letters in the feedback of the guess is the same as the position of the letters in the feedback of the past guess

                    matching_letters = False

                    for i in range(self.env.word_length):#for each letter in the guess

                        if candidate_feedback[i] > 0:#if the letter has been valued 1 or 2 in the feedback add it to the array
                            alphabet_array_candidate[ord(guesses[-1][i]) - ord('a')] += 1


                    for i in range(len(alphabet_array_candidate)):#for each letter in the alphabet

                        if alphabet_array_candidate[i] >= alphabet_array_matches_final[i]:#checks if the letter has been valued 1 more times than in the feedback of the guess as many times as in the feedback of the past guesses
                            matching_letters = True
                        else:

                            break

                    if matching_letters:#if the letters match, the word is added to the list of matching words
                        total_matching.append(candidate)

        return total_matching#returns the list of matching words

    def calculate_entropy(self, words=None):#function that calculates the entropy of the guesses
        Entropy_list = []
        if words is None:#if no list of words is provided, the list of allowed words is used
            words= self.env.allowed_words.tolist()

        max_entropy = -np.inf#maximum entropy
        best_guess = None
        word_total = len(words)

        for i in tqdm(range(len(words)), disable=self.silent):#for each word in the list of allowed words
            candidate = words[i]
            pattern_counts = defaultdict(int)#dictionary that stores the number of times each feedback pattern has been seen

            for word in words:#for each word in the list of words
                match = check_letters_maches(candidate, word)#get the feedback of the guess and the word
                feedback_tuple = tuple(match)#convert the feedback to a tuple

                pattern_counts[feedback_tuple] += 1#add the feedback to the dictionary

            entropy = 0.0
            for count in pattern_counts.values():#update the entropy calculation by adding the entropy of the candidate
                p = count / word_total
                if p > 0:
                    entropy += p * np.log2(1 / p)
            Entropy_list.append((entropy, candidate))#add the entropy of the candidate to the list of entropies

            if entropy > max_entropy:#update the maximum entropy and the best guess
                max_entropy = entropy
                best_guess = candidate

        return best_guess, max_entropy, Entropy_list #return the best guess, the maximum entropy and the list of entropies
    def save_starting_entropy_list(self):#function that saves the entropy list for the first guess to a file (stay the same for every first guess for a given word list)
        best_guess, max_entropy, Entropy_list = self.calculate_entropy()
        sorted_Entropy_list = sorted(Entropy_list, key=lambda x: x[0],reverse=True)
        with open(f'{self.env.path_name}_Entropy_list.pkl', 'wb') as f:
            pickle.dump(sorted_Entropy_list, f)
        return sorted_Entropy_list

    def load_starting_entropy_list(self):#function that loads the entropy list for the first guess from a file
        with open(f'{self.env.path_name}_Entropy_list.pkl', 'rb') as f:
            sorted_Entropy_list = pickle.load(f)
        return sorted_Entropy_list
    def print_Entropy_list(self):#function that prints the entropy list
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

    def get_stats(self, save_path=None):#function that plays the game for all the words in the list of allowed words and saves the statistics to a file
        if save_path and os.path.exists(f'{save_path}.pkl'):#if the file exists, the statistics are loaded from the file
            print(f"Loading stats from {save_path}.pkl")
            with open(f'{save_path}.pkl', 'rb') as f:
                stats=pickle.load(f)
            print(stats)
            return stats
        results = {}#dictionary that stores the results of the games
        # Changed to start from 0 to include first-try wins
        tries_distribution = {i: 0 for i in range(0, self.env.max_tries + 2)}#dictionary that stores the distribution of the number of tries (+2 to include first-try wins at 1 (position)
                                                                                # and max-tries losses)
        total_games = 0
        wins = 0

        for word in tqdm(self.env.allowed_words):#for each word in the list of allowed words
            win, tries = self.play_max_entropy(target_word=word)#play the game
            results[word] = {"win": win, "tries": tries}#store the result of the game
            total_games += 1#update the number of games
            if win:
                wins += 1
                tries_distribution[tries] += 1#update the distribution of the number of tries
            else:
                tries_distribution[self.env.max_tries + 1] += 1#

        self.game_stats = {#store the statistics
            "results": results,
            "total_games": total_games,
            "wins": wins,
            "win_rate": wins / total_games * 100,
            "tries_distribution": tries_distribution
        }

        if save_path is not None:#save the statistics to a file
            print(f"Saving stats to {save_path}.pkl")
            with open(f'{save_path}.pkl', 'wb') as f:
                pickle.dump(self.game_stats, f)
        print(self.game_stats)
        return self.game_stats





em=Entropy_maximizer(Environment('wordle-nyt-allowed-guesses-update-12546.txt'),silent=False)

#em.play_max_entropy()

#em.get_stats('wordle_stats')

