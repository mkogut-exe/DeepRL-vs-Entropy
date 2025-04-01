import string

def missing_start_letters(file_path):
    with open(file_path, 'r') as f:
        words = f.read().splitlines()
    # Collect the first letter of each word (lowercase)
    first_letters = {word[4].lower() for word in words if word}
    # Full set of lowercase letters
    all_letters = set(string.ascii_lowercase)
    # Calculate missing letters
    missing_letters = sorted(all_letters - first_letters)
    return missing_letters

if __name__ == "__main__":
    file_path = "reduced_set.txt"
    missing = missing_start_letters(file_path)
    print("Missing start letters:", missing)