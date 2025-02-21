from Wordle import Environment
import random
class Matching_roulette:#class that plays the game bu just choosing the word form the list of matching words at random
    def __init__(self, path):
        self.env = Environment(path)


    def guess(self):
        while not self.env.end:
            matches = self.env.get_matches()
            len_matches = len(matches)
            if len_matches != 0:
                self.env.guess(random.choice(self.env.allowed_words))
            else:
                break
            guess = self.env.get_guess()
            self.env.guess(guess)

    def get_average(self, num_games):
        total_tries = 0
        wins=0
        for _ in range(num_games):
            self.env.reset()
            self.guess()
            total_tries += self.env.try_count
            if self.env.win:
                wins += 1
        tries_avg = total_tries / num_games
        wins_avg = wins / num_games
        return tries_avg, wins_avg
Matching_roulette("wordle-nyt-allowed-guesses-update-12546.txt").get_average(5000)