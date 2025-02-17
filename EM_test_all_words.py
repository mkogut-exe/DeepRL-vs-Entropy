import os
from Wordle import Environment
from Entropy_Maximizer import Entropy_maximizer

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    # Use absolute path for the input file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'wordle-nyt-allowed-guesses-update-12546.txt')

    em = Entropy_maximizer(Environment(input_file), silent=True)
    em.get_stats('wordle_stats')