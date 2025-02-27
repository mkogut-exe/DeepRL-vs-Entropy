from Wordle import Environment, check_letters_maches
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
from multiprocessing import Pool, cpu_count



class Entropy_maximizer:#class that maximizes the entropy of the guesses and usses it to guess the word
    def __init__(self, env:Environment, silent=False):
        self.env = env #create an instance of the Environment class
        self.silent = silent#variable that determines if the game is played in silent mode (no print statements)
        self.sorted_Entropy_list = None
        if not silent:
            print(f'Word to guess: {self.env.word}')





    def maximizer_guess(self, guess, debug=False):#function that makes a guess based on the entropy of the guesses
        matches = self.env.guess(guess)
        if not self.silent:
            print(f'Guess: {guess}')
        if debug:
            print(self.env.guesses)
            print(self.env.guess_maches)
            print("maximizer_guess")


    def play_max_entropy(self,target_word=None, debug=False):#function that plays the game using the max entropy of the guesses

        if target_word is None:#reset the game if no target word is provided random one is chosen
            self.env.reset()
        else:
            self.env.reset(target_word)
        if not self.silent:
            print(self.env.word)
        matches = self.env.allowed_words#list of words that match the feedback of the guesses
        while self.env.try_count < self.env.max_tries:#while the number of tries is less than the maximum number of tries

            entropies = None

            if self.env.win:
                if not self.silent:
                    print('Win!')

                break
            else:
                if not self.silent:
                    print(f'\nTurn: {self.env.try_count + 1}')

            if self.env.try_count == 0:#if it's the first guess use the precalculated entropy list form .pkl file if it exists otherwise create one
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

                self.maximizer_guess(sorted_Entropy_list[0][1])#make the guess based on the entropy of the initial word list
                if not self.silent:
                    print(f'    Guess: {sorted_Entropy_list[0][1]}, Entropy: {sorted_Entropy_list[0][0]}')
            else:
                if self.env.try_count >= 1:#if it's not the first guess
                    matches=self.env.find_matches(word_list=matches,silent=self.silent)#find the words that match the feedback of the guesses
                    guess, entropy, entropies = self.calculate_entropy(matches)#calculate the entropy of the matching words

                    if debug:
                        sorted_entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
                        if not self.silent:
                            for i, (entropy, word) in enumerate(sorted_entropies):
                                print(f'{i + 1}. {word}: {entropy}')
                    self.maximizer_guess(guess)#make the guess based on the entropy of the matching words
                    if not self.silent:
                        print(f'    Guess: {guess}, Entropy: {entropy}')

        if not self.env.win:#if the game last longer than the maximum number of tries, the game is lost
            if not self.silent:
                print('\nLoss!')
                print(f'The word was: {self.env.word}')
                print(f'Matching words with entropies at the end of te game:\n')
                sorted_entropies=sorted(entropies, key=lambda x: x[0],reverse=True)
                for i, (entropy, word) in enumerate(sorted_entropies):#print the matching words and their entropies for the last guess
                    print(f'{i + 1}. {word}: {entropy}')
        return self.env.win, self.env.try_count#return the result of the game and the number of tries



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

    def load_starting_entropy_list(self):  # function that loads the entropy list for the first guess from a file
        if self.sorted_Entropy_list is None:
            with open(f'{self.env.path_name}_Entropy_list.pkl', 'rb') as f:
                self.sorted_Entropy_list = pickle.load(f)
        return self.sorted_Entropy_list
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

    # function that plays the game for all the words in the list of allowed words

    def get_stats(self, save_path=None): #IF NO SAVED STATS USE ONLY via EM_stats.py (due to multiprocessing)
        if save_path and os.path.exists(f'{save_path}.pkl'):
            print(f"Loading stats from {save_path}.pkl")
            with open(f'{save_path}.pkl', 'rb') as f:
                stats = pickle.load(f)
            print(stats)
            return stats

        results = {}
        tries_distribution = {i: 0 for i in range(0, self.env.max_tries + 2)}
        total_games = 0
        wins = 0

        # Process words in smaller batches
        batch_size = 20
        n_processes = min(cpu_count(), 14)  # Limit number of processes

        # Create word batches
        word_batches = []
        for i in range(0, len(self.env.allowed_words), batch_size):
            batch = self.env.allowed_words[i:i + batch_size]#split the list of allowed words into batches
            word_batches.append([(word, self.env.path_name) for word in batch])# for each word in the batch, add the word and the name of the file with the allowed words to the list

        with Pool(processes=n_processes) as pool:#create a pool of processes
            try:
                for batch in tqdm(word_batches, desc="Processing words"):#for each batch of words
                    # Process batch in parallel
                    batch_results = pool.map_async(process_word_parallel, batch).get(timeout=60)#process the batch of words in parallel

                    # Update statistics
                    for word, win, tries in batch_results:
                        results[word] = {"win": win, "tries": tries}
                        total_games += 1
                        if win:
                            wins += 1
                            tries_distribution[tries] += 1
                        else:
                            tries_distribution[self.env.max_tries + 1] += 1

                    # Save intermediate results
                    if save_path:
                        intermediate_stats = {
                            "results": results,
                            "total_games": total_games,
                            "wins": wins,
                            "win_rate": wins / total_games * 100 if total_games > 0 else 0,
                            "tries_distribution": tries_distribution
                        }
                        with open(f'{save_path}_intermediate.pkl', 'wb') as f:
                            pickle.dump(intermediate_stats, f)

            except TimeoutError:
                print("Processing timeout. Saving partial results.")
            finally:
                pool.close()
                pool.join()

        self.game_stats = {
            "results": results,
            "total_games": total_games,
            "wins": wins,
            "win_rate": wins / total_games * 100 if total_games > 0 else 0,
            "tries_distribution": tries_distribution
        }

        if save_path:
            print(f"Saving stats to {save_path}.pkl")
            with open(f'{save_path}.pkl', 'wb') as f:
                pickle.dump(self.game_stats, f)
        print(self.game_stats)
        return self.game_stats


def process_word_parallel(args):
    word, path_name = args
    # Add .txt extension if not present
    if not path_name.endswith('.txt'):
        path_name = f'{path_name}.txt'
    local_em = Entropy_maximizer(Environment(path_name), silent=True)
    win, tries = local_em.play_max_entropy(target_word=word)
    return word, win, tries

em=Entropy_maximizer(Environment('wordle-nyt-allowed-guesses-update-12546.txt'), silent=False)

print(em.env.allowed_words)
