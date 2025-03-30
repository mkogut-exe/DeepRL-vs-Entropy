# DeepRL-vs-Entropy

## Wordle Game Playing Algorithms:
### Wordle.py 
Environment for Wordle game

### Matching_roulette.py
Wordle agent using matching roulette algorithm to guess the word (just randomly guessing form the matching words)

### Entropy_Maximizer.py
Wordle agent using entropy maximization to guess the word
(needs EM_test_all_words.py in order to run in parallel)


### DeepRL_word_vector.py 
Wordle agent using deep reinforcement learning

#### input:
word vector representation of the all matching words

### output:
outputs guessed word


### DeepRL_letter_guesser.py
Wordle agent using deep reinforcement learning using individual letter guesses to find the word (by combining their probabilities and looking at the list of valid words ot see which one is the most probable that way)

#### input: 
the game state which is represented with a 5X26 binary matrix of all possible letters at each position
#### output: 
matix of probabilities of each letter at each position (5X26) and the guessed word (5 letters)
(which is then procesed to get a probability of each word made form these letter combinations)

### DeepRL_letter_guesser_win_reward.py 
same as the normal DeepRL_letter_guesser but with a reward for winning the game

#### input: 
the game state which is represented with a 5X26 binary matrix of all possible letters at each position
#### output: 
matix of probabilities of each letter at each position (5X26) and the guessed word (5 letters)
(which is then procesed to get a probability of each word made form these letter combinations)

### DeepRL_autoregressive_letter_guesser.py 
Wordle agent using autoregresive deep reinforcement learning where each of the letters is guessed one by one by 5 Actor-Critic pairs each getting the guessed letter form the previous one as input (1st one gets input of just the possible letters matrix, 2nd one gets the output of the first one and the possible letters matrix, etc.)

#### input: 
the game state which is represented with a 5X26 binary matrix of all possible letters at each position +
the guessed letter from the previous actor-critic pair represented with a 1x26 vector, which is then combined with the possible letters matrix

## Helper Code:
### RL_Plotter.py
plots the results of the training of the agent (reward, loss, etc.)

### EM_test_all_words.py
Runs Entropy maximization on all words using parallel processing and saves the results in a file 