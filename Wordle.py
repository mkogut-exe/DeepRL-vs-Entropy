import numpy as np

def get_data(path):#Helper function to get the data from the .txt file
    data = np.genfromtxt(path, dtype=str, delimiter=' ')
    return data

def replace_value(array, old_value, new_value):#Helper function, replaces the old value with the new value in the array
    array[array == old_value] = new_value
    return array

def check_letters_maches(guess, word): #checks if the letters in the guess are in the word and provides feedback 2 if the letter is in the correct position,
                                        # 1 if the letter is in the word but not in the correct position
    result = np.zeros(len(guess), dtype=int)
    letter_count = {letter: word.count(letter) for letter in set(word)}
    correct_position = 0
    in_word = 0
    for i, letter in enumerate(guess):
        if letter in word:
            if letter == word[i] and letter_count[letter] > 0:
                result[i] = 2
                correct_position+=1
                letter_count[letter] -= 1
            elif letter_count[letter] > 0:
                result[i] = 1
                in_word+=1
                letter_count[letter] -= 1
    return result,correct_position,in_word

class Environment:#class that simulates the wordle game
    def __init__(self, allowed_words_path, max_tries=6,word_length=5, word_test=None):
        self.path_name = allowed_words_path.split('.')[0]#name of the file with the allowed words
        self.allowed_words = get_data(allowed_words_path)#list of words that are allowed to be used in the game
        self.max_tries = max_tries#maximum number of tries
        self.try_count = 0 #number of tries
        self.win = False
        self.end = False
        self.word_length = word_length#length of the words that are allowed to be used
        self.guesses = np.array([]) #list of guesses
        self.guess_maches = []#list of the feedback of the guesses
        self.correct_position = 0
        self.in_word = 0


        if word_test is not None:#if a word is provided, the word is set to the provided word otherwise a random word is chosen from the allowed words
            self.word = word_test
        else:
            self.word = np.random.choice(self.allowed_words)

    def check_letters_in_word(self, guess):#checks guess with the word selected at the start of the game
        return check_letters_maches(guess, self.word)
    
    def find_matches(self,word_list=None, debug=False,silent=True):#function that finds the words that match the feedback of the guesses
        guesses = self.guesses.tolist()#list of past guesses

        if word_list is None:#if no list of words is provided, the list of allowed words is used
            word_list = self.allowed_words
        total_matching = []
        if len(guesses) == 0:
            return word_list.tolist()

        zero_letters=set()#set of letters that have been valued 0 in the feedback of the guesses
        present_letters=set()#set of letters that are in the target word (feedback of the guesses)
        absent_letters=set()#set of letters that are not in the target word (feedback of the guesses)

        alphabet_array_matches_final = np.zeros(26, dtype=int)#array that stores the number of times each letter has been valued 1 in the feedback in all past guesses

        for t in range(len(self.guess_maches)):#for each past guess
            alphabet_array_matches = np.zeros(26, dtype=int)#temporary array that stores the number of times each letter has been valued 1 in the feedback of the guess

            for i in range(self.word_length):#for each letter in the guess
                if self.guess_maches[t][i] >= 1:#if the letter has been valued 1 or 2 in the feedback
                    alphabet_array_matches[ord(guesses[t][i]) - ord('a')] += 1
                    present_letters.add(guesses[t][i])#add the letter to the set of present letters


                elif self.guess_maches[t][i] == 0 and guesses[t][i] not in present_letters:#if the letter has been valued 0 in the feedback and is not in the set of present letters
                    zero_letters.add(guesses[t][i])#add the letter to the set of zero letters

            for i in range(len(alphabet_array_matches)):
                alphabet_array_matches_final[i] = max(alphabet_array_matches_final[i], alphabet_array_matches[i])#update the array that stores the number of times each letter has been valued 1 in the feedback in all past guesses

        if debug:
            print(f'zero_letters: {zero_letters}')
            print(f'present_letters: {present_letters}')

        absent_letters=zero_letters - present_letters #set of letters that are not in the target word (feedback of the guesses)
        if not silent:
            print(f' absent_letters: {absent_letters}')
        for candidate in word_list:#for each word in the list of suspected matching words
            if absent_letters.isdisjoint(candidate):#checks if the word has any of the letters that are not in the target word and skips the word if it has
                alphabet_array_candidate = np.zeros(26, dtype=int)

                candidate_feedback,_,_ = check_letters_maches(guesses[-1], candidate)

                #get the guaranteed positions feedback of the guess and candidates
                candidate_correct_position_matches = replace_value(candidate_feedback.copy(), 1, 0)
                if len(self.guess_maches) > 0:
                    maximizer_correct_position_matches = replace_value(self.guess_maches[-1].copy(), 1, 0)
                else:
                    maximizer_correct_position_matches = np.zeros(self.word_length, dtype=int)
                    print(f"ERROR NO MATCH, mathces: {self.guess_maches}, last_guess: {self.guesses},target word: {self.word}")
                    break



                if np.array_equal(candidate_correct_position_matches,maximizer_correct_position_matches):#checks if position of the letters in the feedback of the guess is the same as the position of the letters in the feedback of the past guess

                    matching_letters = False

                    for i in range(self.word_length):#for each letter in the guess

                        if candidate_feedback[i] > 0:#if the letter has been valued 1 or 2 in the feedback add it to the array
                            alphabet_array_candidate[ord(guesses[-1][i]) - ord('a')] += 1


                    for i in range(len(alphabet_array_candidate)):#for each letter in the alphabet

                        if alphabet_array_candidate[i] >= alphabet_array_matches_final[i]:#checks if the letter has been valued 1 more times than in the feedback of the guess as many times as in the feedback of the past guesses
                            matching_letters = True
                        else:

                            break

                    if matching_letters:#if the letters match, the word is added to the list of matching words
                        total_matching.append(candidate)
        #number of present letters in the target word


        return total_matching#returns the list of matching words

    def guess(self, guess):
        self.try_count += 1
        if self.try_count > self.max_tries:#if the number of tries is greater than the maximum number of tries, the game is over
            self.end = True
            self.win = False
            return np.full(5, -1)
        self.guesses = np.append(self.guesses, guess)#adds the guess to the list of past guesses
        matches,correct_position,in_word=self.check_letters_in_word(guess)#checks the guess with the word
        self.correct_position = correct_position#number of letters in the correct position
        self.in_word = in_word#number of letters in the word but not in the correct position
        if np.array_equal(matches, np.full(5, 2)):#if the guess is correct, the game is won
            self.win = True
            self.end = True
            return matches
        self.guess_maches.append(matches)#adds the feedback of the guess to the list of past feedbacks

        self.win = False
        return matches

    def reset(self, word_test=None, allowed_words_path=None):#resets the game
        self.try_count = 0
        self.win = False
        self.end = False
        if allowed_words_path is not None:
            self.allowed_words = get_data(allowed_words_path)#if a new list of allowed words is provided, the list is updated
        self.guesses = np.array([])
        self.guess_maches = []
        if word_test is not None:#if a new word is provided, the word is updated else it's again chosen randomly from the allowed words
            self.word = word_test
        else:
            self.word = np.random.choice(self.allowed_words)
def manual(path):#manual game mode
    env = Environment(path)
    print(f'Mistery word: {env.word}')
    while env.try_count < env.max_tries:
        guess = input(f'Enter your guess (remaining tries: {env.max_tries-env.try_count}): ')
        if guess not in env.allowed_words:
            print('Invalid guess')
            continue
        result = env.guess(guess)
        print(result)
        if np.all(result == 2):
            print('You won!')
            break
    if env.try_count >= env.max_tries:
        print('You lost! The word was: ', env.word)


#manual('wordle-nyt-allowed-guesses-update-12546.txt')