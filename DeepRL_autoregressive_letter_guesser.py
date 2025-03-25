import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Wordle import Environment
import random
from tqdm import tqdm
import pickle
import os
import csv
from multiprocessing import Pool, cpu_count
import torch.nn.utils.prune as prune
import datetime

# Check torch version and CUDA availability
torch_version = torch.__version__
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print(f'Torch version: {torch_version}, CUDA availability: {cuda_available}, Device: {device}')

# Set random seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def create_model_id(epochs, actor_repetition, critic_repetition, actor_network_size, learning_rate, batch_size):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"_{timestamp}_ARLGv1_epo-{epochs}_AR-{actor_repetition}_CR-{critic_repetition}_AS-{actor_network_size}-Lr-{learning_rate}-Bs-{batch_size}"
    # - Rv - Version of the model with no win reward
    # - epo: Number of epochs
    # - AR: Actor repetition count
    # - CR: Critic repetition count
    # - AS: Actor network size
    # - Lr: Learning rate
    # - Bs: Batch size


class Actor:
    def __init__(self, env: Environment, batch_size=256, discount=0.99, epsilon=0.1, learning_rate=1e-4,
                 actor_repetition=15, critic_repetition=5, prune=False, prune_amount=0.1, prune_freq=1000,
                 sparsity_threshold=0.1, random_batch=False, sample_size=256):
        self.env = env
        self.discount = discount
        self.batch_size = batch_size
        self.actor_repetition = actor_repetition
        self.critic_repetition = critic_repetition
        self.epsilon = epsilon
        self.model_id = ''
        self.sparsity_threshold = sparsity_threshold
        self.prune_amount = prune_amount
        self.prune_freq = prune_freq
        self.prune = prune
        self.random_batch = random_batch
        self.sample_size = sample_size
        self.learning_rate = learning_rate

        self.allowed_words_length = len(self.env.allowed_words)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.env.allowed_words)}
        self.allowed_words_tensor = torch.tensor([self.word_to_idx[w] for w in self.env.allowed_words], device=device)

        # Actor network for autoregressive prediction
        # Input: game state (5x26) + previous letter predictions (0-4 positions, one-hot encoded)
        # For the first letter, we'll have just the game state
        # For the second letter, we'll have game state + first letter (one-hot)
        # And so on...

        # Base input size is the game state
        base_input_size = self.env.word_length * 26

        # Actor network for autoregressive prediction
        # For each position, we'll have a separate network that takes:
        # - Current game state (5x26)
        # - Previous letter predictions (0 to 4 positions, one-hot encoded)
        self.actor = nn.ModuleList([
            # Position-specific networks
            nn.Sequential(
                # Input: game state + previous letter context (one-hot encoded)
                nn.Linear(base_input_size + pos * 26, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                # Output: probability distribution over 26 letters for this position
                nn.Linear(256, 26)
            ).to(device)
            for pos in range(self.env.word_length)
        ])

        # Critic network - evaluates the state only (not position-specific)
        self.critic = nn.Sequential(
            nn.Linear(self.env.word_length * 26, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        ).to(device)

        # Create optimizers for all position networks
        self.optimizer_actor = optim.Adam(
            [param for network in self.actor for param in network.parameters()],
            lr=self.learning_rate
        )
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.stats = {
            'total_games': 0,
            'wins': 0,
            'win_rate': 0,
            'tries_distribution': {i: 0 for i in range(0, 8)},
            'reward_history': {},  # This will store episode -> reward mapping
            'results': {}
        }

    def save_model(self, path):
        torch.save({
            'actor': [net.state_dict() for net in self.actor],
            'critic': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        for i, state_dict in enumerate(checkpoint['actor']):
            self.actor[i].load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic'])

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        return self.stats

    def save_training_metrics(self, episode, actor_loss, critic_loss, win_rate, avg_reward,
                              metrics_file='training_metrics'):
        full_path = f"{metrics_file}{self.model_id}.csv"
        file_exists = os.path.isfile(full_path)

        with open(full_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate', 'Reward'])
            writer.writerow([episode, actor_loss, critic_loss, win_rate, avg_reward])

    def state(self):
        state = self.env.get_letter_possibilities_from_matches(self.env.find_matches())
        return torch.FloatTensor(state.flatten()).to(device)  # Flatten the 5x26 array

    def act_word(self):
        with torch.no_grad():
            state = self.state()
            word_letters = []
            letter_probs = []

            # Generate each letter autoregressively
            for position in range(self.env.word_length):
                # For each position, we need the game state plus previous letter predictions
                if position == 0:
                    # First letter: just the game state
                    input_vector = state
                else:
                    # Later letters: state + one-hot encoding of previous letters
                    previous_letters_onehot = torch.zeros(position * 26, device=device)
                    for prev_pos, letter_idx in enumerate(word_letters):
                        previous_letters_onehot[prev_pos * 26 + letter_idx] = 1.0
                    input_vector = torch.cat([state, previous_letters_onehot])

                # Get logits for this position from the position-specific network
                logits = self.actor[position](input_vector)

                # Apply mask based on letter possibilities
                state_reshaped = state.view(self.env.word_length, 26)
                position_mask = state_reshaped[position]
                masked_logits = logits + (position_mask - 1) * 1e8

                # Get probability distribution
                probs = torch.softmax(masked_logits, dim=0)

                # Sample letter based on probabilities
                if torch.sum(probs) <= 1e-10:
                    # If all probabilities are essentially zero, pick a valid letter uniformly
                    valid_indices = torch.where(position_mask > 0)[0]
                    if len(valid_indices) == 0:
                        # Fallback if no valid letters (shouldn't happen)
                        letter_idx = random.randrange(26)
                    else:
                        letter_idx = valid_indices[random.randrange(len(valid_indices))].item()
                    letter_prob = 1.0 / len(valid_indices) if len(valid_indices) > 0 else 1e-8
                else:
                    letter_idx = torch.multinomial(probs, 1).item()
                    letter_prob = probs[letter_idx].item()

                word_letters.append(letter_idx)
                letter_probs.append(letter_prob)

            # Convert letter indices to word
            generated_word = ''.join(chr(idx + ord('a')) for idx in word_letters)

            # Find closest valid word
            if generated_word in self.word_to_idx:
                action = self.word_to_idx[generated_word]
                action_prob = torch.prod(torch.tensor(letter_probs)).item()
            else:
                # If generated word isn't valid, find closest valid word
                matching_words = self.env.find_matches()
                if not matching_words:
                    # Fallback if no matching words
                    return random.randrange(len(self.env.allowed_words)), torch.tensor(1e-8, device=device)

                # Find word with most matching letters
                best_match = None
                best_score = -1
                for word in matching_words:
                    score = sum(1 for i, c in enumerate(word) if ord(c) - ord('a') == word_letters[i])
                    if score > best_score:
                        best_score = score
                        best_match = word

                action = self.word_to_idx[best_match]
                # Approximate probability - product of probabilities of matching letters
                action_prob = 1e-8

            return action, torch.tensor(action_prob, device=device)

            # Convert to tensor and normalize
            word_probs = torch.tensor(word_probs, device=device)

            # Handle case where all probs are 0
            if word_probs.sum() <= 1e-10:
                chosen_idx = random.choice(indices)
                return chosen_idx, torch.tensor(1e-8, device=device)

            word_probs = word_probs / word_probs.sum()

            # Sample word according to probabilities
            chosen_idx_in_list = torch.multinomial(word_probs, 1).item()
            action = indices[chosen_idx_in_list]

            return action, word_probs[chosen_idx_in_list]

    def batch_update(self, states, actions, rewards, next_states, old_action_probs_selected, dones):
        random = self.random_batch

        states = torch.stack(states)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.stack(next_states)
        old_action_probs_selected = torch.tensor(old_action_probs_selected, device=device, dtype=torch.float32)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)

        # Critic update - always use full batch
        with torch.no_grad():
            current_values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            target_values = rewards + self.discount * next_values * (1 - dones)
            td_errors = target_values - current_values

        # Update critic
        critic_losses = []
        for _ in range(self.critic_repetition):
            self.optimizer_critic.zero_grad()
            current_values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(current_values, target_values.detach())
            critic_loss.backward()
            self.optimizer_critic.step()
            critic_losses.append(critic_loss.item())

        # Actor update
        actor_losses = []
        td_errors_detached = td_errors.detach()
        batch_size = len(states)

        for _ in range(self.actor_repetition):
            self.optimizer_actor.zero_grad()

            if random and batch_size > self.sample_size:
                # Split the batch into games based on done flags
                done_indices = (dones == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                game_indices = []
                start = 0
                for end in done_indices:
                    end = end.item()
                    game_indices.append((start, end))
                    start = end + 1
                if start < len(dones):
                    game_indices.append((start, len(dones) - 1))

                num_games = len(game_indices)
                if num_games == 0:
                    indices = torch.arange(batch_size, device=device)
                else:
                    shuffled_indices = torch.randperm(num_games, device=device)
                    selected_transitions = []
                    total = 0
                    for i in shuffled_indices:
                        game_start, game_end = game_indices[i]
                        game_length = game_end - game_start + 1
                        if total + game_length > self.sample_size and total > 0:
                            continue
                        selected_transitions.extend(range(game_start, game_end + 1))
                        total += game_length
                        if total >= self.sample_size:
                            break
                    indices = torch.tensor(selected_transitions, device=device)

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_td_errors = td_errors_detached[indices]
                batch_old_probs = old_action_probs_selected[indices]
            else:
                batch_states = states
                batch_actions = actions
                batch_td_errors = td_errors_detached
                batch_old_probs = old_action_probs_selected

            # Get new action probabilities - process each position separately
            mini_batch_size = batch_states.size(0)
            all_letter_indices = torch.zeros((mini_batch_size, self.env.word_length), dtype=torch.long, device=device)

            # Convert word indices to letter indices
            for i, action_idx in enumerate(batch_actions):
                word = self.env.allowed_words[action_idx]
                all_letter_indices[i] = torch.tensor([ord(c) - ord('a') for c in word], device=device)

            # Process each position's probabilities
            all_probs = []

            for position in range(self.env.word_length):
                # For first position, just use the state
                if position == 0:
                    inputs = batch_states
                else:
                    # For later positions, add previous letter predictions
                    prev_letters = all_letter_indices[:, :position]
                    prev_onehot = torch.zeros(mini_batch_size, position * 26, device=device)

                    # Create one-hot encodings of previous letters
                    for batch_idx in range(mini_batch_size):
                        for prev_pos, letter_idx in enumerate(prev_letters[batch_idx]):
                            prev_onehot[batch_idx, prev_pos * 26 + letter_idx] = 1.0

                    inputs = torch.cat([batch_states, prev_onehot], dim=1)

                # Get probabilities from position network
                pos_logits = self.actor[position](inputs)

                # Apply mask based on letter possibilities
                states_reshaped = batch_states.view(mini_batch_size, self.env.word_length, 26)
                position_masks = states_reshaped[:, position]
                masked_logits = pos_logits + (position_masks - 1) * 1e10

                pos_probs = torch.softmax(masked_logits, dim=1)
                all_probs.append(pos_probs)

            # Stack probabilities and select the ones for the chosen letters
            all_probs = torch.stack(all_probs, dim=1)  # [batch_size, word_length, 26]

            batch_pos_indices = torch.arange(self.env.word_length, device=device).unsqueeze(0).expand(mini_batch_size,
                                                                                                      -1)
            batch_indices = torch.arange(mini_batch_size, device=device).unsqueeze(1).expand(-1, self.env.word_length)

            selected_probs = all_probs[batch_indices, batch_pos_indices, all_letter_indices]
            new_action_probs_selected = selected_probs.prod(dim=1)

            importance_ratios = new_action_probs_selected / (batch_old_probs + 1e-10)
            clipped_ratios = torch.clamp(importance_ratios, 1 - self.epsilon, 1 + self.epsilon)
            loss = torch.min(
                importance_ratios * batch_td_errors,
                clipped_ratios * batch_td_errors
            )
            actor_loss = -loss.mean()

            actor_loss.backward()
            self.optimizer_actor.step()
            actor_losses.append(actor_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    def train(self, epochs=500, print_freq=50, autosave=False, append_metrics=False, prune_amount=0.1, prune_freq=1000,
              sparsity_threshold=0.1, prune=False):
        print("Training...")
        self.prune_amount = prune_amount
        self.prune_freq = prune_freq
        self.sparsity_threshold = sparsity_threshold
        self.prune = prune
        self.model_id = create_model_id(epochs=epochs, actor_repetition=self.actor_repetition,
                                        critic_repetition=self.critic_repetition, actor_network_size='2x256',
                                        learning_rate=self.learning_rate, batch_size=self.batch_size)
        total_wins = 0
        batch_losses_actor = []
        batch_losses_critic = []
        episode_rewards = []  # Track rewards for each episode

        # Initialize replay buffer
        replay_buffer = []

        # Create or append to metrics file
        with open(f'training_metrics{self.model_id}.csv', 'a' if append_metrics else 'w', newline='') as f:
            writer = csv.writer(f)
            if not append_metrics:
                writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate', 'Reward'])

        for episode in tqdm(range(epochs)):
            self.env.reset()
            state = self.state()
            last_correct = 0
            last_in_word = 0
            episode_total_reward = 0  # Track total reward for this episode

            for round in range(self.env.max_tries):
                action, old_prob = self.act_word()
                matches = self.env.guess(self.env.allowed_words[action])
                next_state = self.state()
                done = self.env.end

                # Calculate reward
                correct_position = self.env.correct_position
                in_word = self.env.in_word

                # Ensure improvements can't be negative (should be rare but possible)
                position_improvement = max(0, correct_position - last_correct)
                word_improvement = max(0, in_word - last_in_word)

                reward = 0
                if self.env.win:
                    reward = 10.0
                # Use different reward based on progress
                if position_improvement > 0 or word_improvement > 0:
                    # Good progress - higher reward
                    reward += 1.0 + position_improvement + word_improvement


                episode_total_reward += reward  # Add to episode total

                last_correct = correct_position
                last_in_word = in_word

                # Add transition to replay buffer
                replay_buffer.append((state, action, reward, next_state, old_prob, done))

                # Process in batches when buffer reaches batch size
                if len(replay_buffer) >= self.batch_size:
                    batch = replay_buffer[:self.batch_size]
                    replay_buffer = replay_buffer[self.batch_size:]

                    states, actions, rewards, next_states, old_probs, dones = zip(*batch)
                    loss_actor, loss_critic = self.batch_update(states, actions, rewards, next_states, old_probs, dones)

                    batch_losses_actor.append(loss_actor)
                    batch_losses_critic.append(loss_critic)

                if done:
                    break

                state = next_state.clone()

            # Update stats
            total_wins += self.env.win
            self.stats['wins'] += self.env.win
            self.stats['total_games'] += 1
            self.stats['tries_distribution'][self.env.try_count] += 1
            self.stats['results'][self.env.word] = {'tries': self.env.try_count, 'win': self.env.win}

            # Store reward history for this episode
            self.stats['reward_history'][episode] = episode_total_reward
            episode_rewards.append(episode_total_reward)

            # Process remaining samples if enough have accumulated
            if len(replay_buffer) >= min(1024, self.batch_size):  # Use smaller mini-batches for leftover data
                mini_batch_size = min(1024, len(replay_buffer))
                batch = replay_buffer[:mini_batch_size]
                replay_buffer = replay_buffer[mini_batch_size:]

                states, actions, rewards, next_states, old_probs, dones = zip(*batch)
                loss_actor, loss_critic = self.batch_update(states, actions, rewards, next_states, old_probs, dones)

                batch_losses_actor.append(loss_actor)
                batch_losses_critic.append(loss_critic)

            # Print stats and save metrics
            if (episode + 1) % print_freq == 0:
                avg_loss_actor = np.mean(batch_losses_actor) if batch_losses_actor else 0
                avg_loss_critic = np.mean(batch_losses_critic) if batch_losses_critic else 0
                win_rate = total_wins / print_freq
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0  # Calculate average reward

                self.save_training_metrics(episode + 1, avg_loss_actor, avg_loss_critic, win_rate, avg_reward)

                print(f"Episode {episode + 1}/{epochs} - Actor Loss: {avg_loss_actor:.4f}, "
                      f"Critic Loss: {avg_loss_critic:.4f}, Win Rate: {win_rate:.4f}, "
                      f"Avg Reward: {avg_reward:.4f}")

                total_wins = 0
                batch_losses_actor = []
                batch_losses_critic = []
                episode_rewards = []  # Reset episode rewards

                if autosave:
                    self.save_model(f'actor_critic_{episode + 1}{self.model_id}.pt')
                self.save_stats(f'actor_critic_stats{self.model_id}.pkl')

        # Process any remaining samples in the buffer at the end
        while len(replay_buffer) >= 1:  # Process remaining data in small batches
            mini_batch_size = min(1024, len(replay_buffer))
            batch = replay_buffer[:mini_batch_size]
            replay_buffer = replay_buffer[mini_batch_size:]

            states, actions, rewards, next_states, old_probs, dones = zip(*batch)
            self.batch_update(states, actions, rewards, next_states, old_probs, dones)

        self.save_model(f'actor_critic_end{self.model_id}.pt')
        self.save_stats(f'actor_critic_stats{self.model_id}.pkl')
        print("Training finished.")

    def continue_training(self, model_path, stats_path=None, epochs=5000, print_freq=500,
                          learning_rate=None, epsilon=None, actor_repetition=None, critic_repetition=None,
                          batch_size=None, random_batch=None, sample_size=None):
        # Load model and stats (unchanged)
        self.load_model(model_path)
        print(f"Loaded model from {model_path}")

        if stats_path and os.path.exists(stats_path):
            self.load_stats(stats_path)
            win_rate = self.stats['wins'] / self.stats['total_games'] if self.stats['total_games'] > 0 else 0
            print(f"Loaded stats from {stats_path}: {self.stats['total_games']} games, win rate: {win_rate:.4f}")

        # Update hyperparameters if specified (unchanged)
        if learning_rate is not None:
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
            print(f"Updated learning rate to {self.learning_rate}")

        if epsilon is not None:
            self.epsilon = epsilon
            print(f"Updated epsilon to {epsilon}")

        if actor_repetition is not None:
            self.actor_repetition = actor_repetition
            print(f"Updated actor repetition to {actor_repetition}")

        if critic_repetition is not None:
            self.critic_repetition = critic_repetition
            print(f"Updated critic repetition to {critic_repetition}")
        if batch_size is not None:
            self.batch_size = batch_size
            print(f"Updated batch size to {batch_size}")
        if random_batch is not None:
            self.random_batch = random_batch
            print(f"Updated random batch to {random_batch}")
        if sample_size is not None:
            self.sample_size = sample_size
            print(f"Updated sample size to {sample_size}")

        # Continue training - now with append_metrics=True to append to existing file
        print(f"Continuing training for {epochs} epochs...")
        self.train(epochs=epochs, print_freq=print_freq, append_metrics=True)

    def run_test(self, path, num_games):
        self.load_model(path)
        total_wins = 0
        total_tries = 0
        for _ in tqdm(range(num_games)):
            self.env.reset()
            while not self.env.end:
                state = self.state()
                action, _ = self.act_word()
                self.env.guess(self.env.allowed_words[action])
            total_tries += self.env.try_count
            if self.env.win:
                total_wins += 1
        avg_tries = total_tries / num_games
        win_rate = total_wins / num_games
        print(f"Average tries: {avg_tries}, Win rate: {win_rate}")
        return avg_tries, win_rate


env = Environment("reduced_set.txt")
A = Actor(env, batch_size=10, epsilon=0.1, learning_rate=1e-5, actor_repetition=10, critic_repetition=2,
          random_batch=True, sample_size=2)
# A.continue_training(model_path='GOOD2_actor_critic_end_Rv2_epo-40000_AR-10_CR-2_AS-8x256-Lr-1e-05-Bs-1024.pt', stats_path='GOOD2_actor_critic_stats_Rv2_epo-40000_AR-10_CR-2_AS-8x256-Lr-1e-05-Bs-1024.pkl', epochs=40000, print_freq=1000, learning_rate=1e-5, epsilon=0.1, actor_repetition=10, critic_repetition=2,batch_size=1024,random_batch=True,sample_size=256)
A.train(epochs=80, print_freq=10, prune=False)

