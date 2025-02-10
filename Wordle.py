import numpy as np

def get_data(path):
    data = np.genfromtxt(path, dtype=str, delimiter=' ')
    return data

def check_letters_maches(guess, word):
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

class Environment:
    def __init__(self, allowed_words_path, max_tries=6,word_length=5):
        self.allowed_words = get_data(allowed_words_path)
        self.max_tries = max_tries
        self.try_count = 0
        self.word_length = word_length
        self.guesses = np.array([])
        self.guess_maches = []
        self.word = 'aahed'#np.random.choice(self.allowed_words)

    def check_letters_in_word(self, guess):
        return check_letters_maches(guess, self.word)

    def guess(self, guess):
        self.try_count += 1
        if self.try_count > self.max_tries:
            return np.full(5, -1)
        self.guesses = np.append(self.guesses, guess)
        matches=self.check_letters_in_word(guess)
        self.guess_maches.append(matches)
        return matches

    def reset(self):
        self.try_count = 0
        self.guesses = np.array([])
        self.word = np.random.choice(self.allowed_words)
def manual(path):
    env = Environment(path)
    print(env.word)
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



