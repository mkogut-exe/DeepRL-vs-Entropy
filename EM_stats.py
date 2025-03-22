import pickle
import numpy as np


def summarize_wordle_stats(file_path='wordle_stats_600.pkl'):
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)

    # Extract main statistics
    total_games = stats['total_games']
    wins = stats['wins']
    win_rate = stats['win_rate']
    tries_dist = stats['tries_distribution']

    # Calculate average tries for wins
    win_tries = []
    for word_data in stats['results'].values():
        if word_data['win']:
            win_tries.append(word_data['tries'])

    avg_tries = np.mean(win_tries) if win_tries else 0

    # Format distribution
    dist_percent = {k: (v / total_games) * 100 for k, v in tries_dist.items()}

    print(f"Total games played: {total_games}")
    print(f"Total wins: {wins}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average tries for wins: {avg_tries:.2f}")
    print("\nTries distribution:")
    for tries, percent in dist_percent.items():
        if tries == 7:  # Assuming max_tries + 1 is 7
            print(f"Failed: {percent:.2f}%")
        else:
            print(f"{tries} tries: {percent:.2f}%")


# Run the summary
summarize_wordle_stats()