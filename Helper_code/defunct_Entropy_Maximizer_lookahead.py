from Wordle import Environment, check_letters_maches
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

def replace_value(array, old_value, new_value):
    array[array == old_value] = new_value
    return array


class Entropy_maximizer:
    def __init__(self, env:Environment):
        self.env = env

        #print(env.word)

    def calculate_entropy(self, words=None):
        Entropy_list = []
        if words is None:
            words= self.env.allowed_words.tolist()
        max_entropy = -np.inf
        best_guess = None
        word_total = len(words)


        for i in tqdm(range(len(words))):
            candidate = self.env.allowed_words[i]
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
            Entropy_list.append(entropy)

            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = candidate

        return best_guess, max_entropy, zip(Entropy_list,self.env.allowed_words)


    def maximizer_guess(self, guess, debug=False):
        self.env.guess(guess)
        if debug:
            print(self.env.guesses)
            print(self.env.guess_maches)
            print("maximizer_guess")


    def Get_entropy(self):
        for i in range(len(self.env.allowed_words)):
            matches=self.env.allowed_words.copy()
            self.env.reset(matches[i])
            print('reset')
            matches=np.delete(matches,i)

            for j in range(self.env.max_tries):
                if len(matches) > 1:
                    print('before', matches, len(matches), j)
                    for i in range(len(matches)-1):
                        match=matches[i]
                        #matches = np.delete(matches.copy(), i)
                        print(matches)
                        self.maximizer_guess(match)

                        matches_next = self.find_matches(matches)

                        number_of_matches = len(matches)
                        print(match,matches_next,number_of_matches, j, i)
                else:
                    break
    def get_entropy(self):
        for i in range(len(self.env.allowed_words)):
            matches = self.env.allowed_words.copy()
            print(matches)
            word = matches[i]
            matches = np.delete(matches, i)
            guesslist = [word]
            while len(matches) > 1:
                self.env.reset(word, matches)
                for match in matches:
                    self.maximizer_guess(match)
                    matches = self.find_matches(matches)
                    print(matches)


    def find_matches(self,word_list=None, debug=False):
        guesses = self.env.guesses.tolist()

        if word_list is None:
            word_list = self.env.allowed_words
        total_matching = []
        if len(guesses) == 0:
            return word_list.tolist()  # Return all words initially

        for candidate in word_list:
            matching_position = False
            alphabet_array_candidate = np.zeros(26, dtype=int)
            alphabet_array_matches_final = np.zeros(26, dtype=int)

            candidate_feedback = check_letters_maches(guesses[-1], candidate)


            candidate_correct_position_matches=replace_value(candidate_feedback.copy(), 1, 0)
            maximizer_correct_position_matches=replace_value(self.env.guess_maches[-1].copy(), 1, 0)

            if np.array_equal(candidate_correct_position_matches,maximizer_correct_position_matches):
                matching_position = True

            else:
                matching_position = False

            for i in range(self.env.word_length):

                if candidate_feedback[i]==1:
                    alphabet_array_candidate[ord(self.env.guesses[-1][i]) - ord('a')] += 1


            for t in range(len(self.env.guess_maches)):
                #print(self.env.guesses)
                #print(self.env.guesses[t])
                alphabet_array_matches = np.zeros(26, dtype=int)
                for i in range(self.env.word_length):
                    if self.env.guess_maches[t][i] == 1:
                        alphabet_array_matches[ord(self.env.guesses[t][i]) - ord('a')] += 1

                for i in range(len(alphabet_array_matches)):

                    alphabet_array_matches_final[i]=max(alphabet_array_matches_final[i], alphabet_array_matches[i])

            #print(candidate, alphabet_array_candidate)
            if np.array_equal(alphabet_array_matches_final, alphabet_array_candidate):
                matching_letter=True

            else:
                matching_letter = False

            if matching_position:

                total_matching.append(candidate)

        if debug:
            print(f'Number of allowed words:{len(self.env.allowed_words)}')
            print(f'Matching:{len(total_matching)}')
            print(f'Guesses:{guesses}')
            print(f'total_matching:{total_matching}')
            print(alphabet_array_matches_final)
        return total_matching

em=Entropy_maximizer(Environment('wordle-nyt-allowed-guesses-update-12546.txt'))
best_guess, max_entropy, Entropy_list_zip = em.calculate_entropy()
Entropy_list = list(Entropy_list_zip)
sorted_Entropy_list = sorted(Entropy_list, key=lambda x: x[0],reverse=True)

# Pickle the sorted Entropy list
with open('sorted_entropy_list.pkl', 'wb') as f:
    pickle.dump(sorted_Entropy_list, f)

# Print the sorted list
for entropy, word in sorted_Entropy_list:
    print(word, entropy)
#em.maximizer_guess("hello")
#print(em.find_matches(True))