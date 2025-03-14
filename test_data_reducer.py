import random


def create_tiny_set(input_file='reduced_set.txt', output_file='thiny_set.txt', num_words=600):
    """
    Select random words from the input file and write them to the output file.

    Args:
        input_file: Source file containing all words
        output_file: Destination file for the random subset
        num_words: Number of words to randomly select

    Returns:
        list: The selected words that were written to the file
    """
    # Read all words from the reduced set
    with open(input_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    # Make sure we don't try to select more words than available
    num_words = min(num_words, len(words))

    # Select random words without replacement
    selected_words = random.sample(words, num_words)

    # Write the selected words to the tiny set file
    with open(output_file, 'w') as f:
        for word in selected_words:
            f.write(word + '\n')

    print(f"Successfully wrote {num_words} random words to {output_file}")
    return selected_words

create_tiny_set()