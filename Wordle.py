import numpy as np

def get_data(path):
    data = np.genfromtxt(path, dtype=str, delimiter=' ')
    return data

def check_letters_maches(guess, word): #checks if the letters in the guess are in the word and provides feedback 2 if the letter is in the correct position,
                                        # 1 if the letter is in the word but not in the correct position
    result = np.zeros(len(guess), dtype=int)
    letter_count = {letter: word.count(letter) for letter in set(word)}

    for i, letter in enumerate(guess):
        if letter in word:
            if letter == word[i] and letter_count[letter] > 0:
                result[i] = 2
                letter_count[letter] -= 1
            elif letter_count[letter] > 0:
                result[i] = 1
                letter_count[letter] -= 1
    return result

class Environment:#class that simulates the wordle game
    def __init__(self, allowed_words_path, max_tries=6,word_length=5, word_test=None):
        self.path_name = allowed_words_path.split('.')[0]#name of the file with the allowed words
        self.allowed_words = get_data(allowed_words_path)#list of words that are allowed to be used in the game
        self.max_tries = max_tries#maximum number of tries
        self.try_count = 0 #number of tries
        self.win = False
        self.word_length = word_length#length of the words that are allowed to be used
        self.guesses = np.array([]) #list of guesses
        self.guess_maches = []#list of the feedback of the guesses
        if word_test is not None:#if a word is provided, the word is set to the provided word otherwise a random word is chosen from the allowed words
            self.word = word_test
        else:
            self.word = np.random.choice(self.allowed_words)

    def check_letters_in_word(self, guess):#checks guess with the word selected at the start of the game
        return check_letters_maches(guess, self.word)

    def guess(self, guess):
        self.try_count += 1
        if self.try_count > self.max_tries:#if the number of tries is greater than the maximum number of tries, the game is over
            return np.full(5, -1)
        self.guesses = np.append(self.guesses, guess)#adds the guess to the list of past guesses
        matches=self.check_letters_in_word(guess)#checks the guess with the word
        if np.array_equal(matches, np.full(5, 2)):#if the guess is correct, the game is won
            self.win = True
            return matches
        self.guess_maches.append(matches)#adds the feedback of the guess to the list of past feedbacks

        self.win = False
        return matches

    def reset(self, word_test=None, allowed_words_path=None):#resets the game
        self.try_count = 0
        self.win = False
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