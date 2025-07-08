import pickle
import numpy as np
import matplotlib.pyplot as plt


def detailed_wordle_stats(file_path=r'C:\Users\mkogut\PycharmProjects\DeepRL-vs-Entropy\Stats\Runs_before_13_03\wordle_stats_first_run.pkl'):
    """Loads stats from a file and prints the detailed distribution of tries."""
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)

    tries_dist = stats['tries_distribution']
    total_games = stats['total_games']

    dist_percent = {k: (v / total_games) * 100 for k, v in tries_dist.items()}
    print("\nTries distribution:")
    # Sort by number of tries
    sorted_tries = sorted(tries_dist.items())

    for tries, count in sorted_tries:
        if tries == 0:
            continue
        percent = dist_percent[tries]
        if tries == 7:  # Assuming max_tries + 1 is 7 for failures
            print(f"Failed: {count} games ({percent:.2f}%)")
        else:
            print(f"{tries} tries: {count} games ({percent:.2f}%)")
    return stats


def summarize_wordle_stats(file_path=r'C:\Users\mkogut\PycharmProjects\DeepRL-vs-Entropy\Stats\Runs_before_13_03\wordle_stats_first_run.pkl'):
    stats = detailed_wordle_stats(file_path)

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

    print(f"Total games played: {total_games}")
    print(f"Total wins: {wins}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average tries for wins: {avg_tries:.2f}")

    # --- Bar graph of tries distribution ---
    tries_labels = []
    tries_values = []
    for tries, count in tries_dist.items():
        if tries == 7:
            tries_labels.append('Failed')
        elif tries != 0:
            tries_labels.append(str(tries))
        else:
            continue
        tries_values.append(count)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(tries_labels, tries_values, color='skyblue')
    plt.xlabel('Number of Tries')
    plt.ylabel('Number of Games')
    plt.title('Wordle Tries Distribution for Entropy Maximizer')
    plt.tight_layout()
    # Add percentage labels on top of bars and set secondary y-axis for percentage
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    total = sum(tries_values)
    percent_values = [v / total * 100 for v in tries_values]
    ax2.set_ylim(ax1.get_ylim()[0] / total * 100, ax1.get_ylim()[1] / total * 100)
    ax2.set_ylabel('Percentage of Games (%)', labelpad=20)  # Add padding to prevent cropping
    ax2.set_yticks([])  # Remove right y-axis numbers
    # Add percentage labels above bars
    for bar, percent in zip(bars, percent_values):
        height = bar.get_height()
        ax1.annotate(f'{percent:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to fit labels
    plt.savefig('tries_distribution_EM.png', format='png')  # Save the figure as PNG
    plt.show()


# Run the summary
summarize_wordle_stats()