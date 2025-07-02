from Wordle import Environment
import random
from tqdm import tqdm
import pickle
class Matching_roulette:#class that plays the game by just choosing the word form the list of matching words at random
    def __init__(self, path):
        self.env = Environment(path)
        self.stats = {
            'total_games': 0,
            'wins': 0,
            'win_rate': 0,
            'tries_distribution': {i: 0 for i in range(1, 8)},
            'results': {}
        }

    def play_one(self):
        while not self.env.end:
            matches = self.env.find_matches()
            self.env.guess(random.choice(matches))
        if self.env.end and (self.env.try_count < self.env.max_tries) and not self.env.win:
            print(self.env.try_count)
            print(matches)
            print(self.env.word)
    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
    def load_stats(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        return self.stats
    def get_average(self, num_games):
        total_tries = 0
        wins=0
        for _ in tqdm(range(num_games)):
            self.env.reset()
            self.play_one()
            total_tries += self.env.try_count
            self.stats['total_games'] += 1
            self.stats['tries_distribution'][self.env.try_count] += 1
            self.stats['results'][self.env.word] = {'tries': self.env.try_count, 'win': self.env.win}


            if self.env.win:
                self.stats['wins'] += 1
                wins += 1
        tries_avg = total_tries / num_games
        wins_avg = wins / num_games
        self.stats['win_rate'] = wins_avg

        self.save_stats('Matching_roulette_stats.pkl')
        print(f"Average tries: {tries_avg}")
        print(f"Win rate: {wins_avg}")
        return tries_avg, wins_avg
    def print_stats(self):
        print(f"Total games: {self.stats['total_games']}")
        print(f"Wins: {self.stats['wins']}")
        print(f"Win rate: {self.stats['win_rate']:.2%}")
        print("Tries distribution:")
        for tries, count in self.stats['tries_distribution'].items():
            print(f"{tries}: {count} games")

MR=Matching_roulette("wordle-nyt-allowed-guesses-update-12546.txt")
MR.load_stats('Matching_roulette_stats_10000_wordle-nyt-allowed-guesses-update-12546.pkl')
MR.print_stats()

#wirate 0.4229