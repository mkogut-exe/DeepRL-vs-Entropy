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


class Actor:
    def __init__(self, env: Environment, batch_size=256, discount=0.99, epsilon=0.1, learning_rate=1e-4,
                 actor_repetition=15, critic_repetition=5):
        self.env = env
        self.discount = discount
        self.batch_size = batch_size
        self.actor_repetition = actor_repetition
        self.critic_repetition = critic_repetition
        self.epsilon = epsilon

        self.allowed_words_length = len(self.env.allowed_words)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.env.allowed_words)}
        self.allowed_words_tensor = torch.tensor([self.word_to_idx[w] for w in self.env.allowed_words], device=device)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.env.word_length*26, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, self.env.word_length*26),
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.env.word_length*26, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.stats = {
            'total_games': 0,
            'wins': 0,
            'win_rate': 0,
            'tries_distribution': {i: 0 for i in range(0, 8)},
            'results': {}
        }

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        return self.stats

    def save_training_metrics(self, episode, actor_loss, critic_loss, win_rate, metrics_file='training_metrics.csv'):
        file_exists = os.path.isfile(metrics_file)

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate'])
            writer.writerow([episode, actor_loss, critic_loss, win_rate])

    def state(self):
        state = self.env.get_letter_possibilities_from_matches(self.env.find_matches())
        return torch.FloatTensor(state.flatten()).to(device)  # Flatten the 5x26 array

    def act_word(self):
        with torch.no_grad():
            state = self.state()
            logits = self.actor(state).view(self.env.word_length, 26)
            state_reshaped = state.view(self.env.word_length, 26)

            # Apply mask for numerical stability
            masked_logits = logits + (state_reshaped - 1) * 1e4
            action_prob = torch.softmax(masked_logits, dim=1)

            # Epsilon-greedy exploration with only matching words
            matching_words = self.env.find_matches()
            if not matching_words:  # Fallback if no matching words
                return random.randrange(len(self.env.allowed_words)), torch.tensor(1e-8, device=device)

            if random.random() < self.epsilon:
                # Random exploration among matching words only
                chosen_word = random.choice(matching_words)
                chosen_idx = self.word_to_idx[chosen_word]
                return chosen_idx, torch.tensor(self.epsilon / len(matching_words), device=device)

            # Calculate probabilities for each matching word
            word_probs = []
            indices = []

            for word in matching_words:
                word_idx = self.word_to_idx[word]
                indices.append(word_idx)

                # Calculate probability as product of letter probabilities
                prob = 1.0
                for pos, letter in enumerate(word):
                    letter_idx = ord(letter) - ord('a')
                    prob *= action_prob[pos, letter_idx].item()

                word_probs.append(prob)

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

    def act_individual_letter(self):
        with torch.no_grad():
            state = self.state()
            logits = self.actor(state).view(self.env.word_length, 26)
            state_reshaped = state.view(self.env.word_length, 26)

            # Apply mask by setting invalid logits to a large negative value
            masked_logits = logits + (state_reshaped - 1) * 1e8  # -inf for invalid
            action_prob = torch.softmax(masked_logits, dim=1)  # Per-position softmax

            # Mask impossible letters
            masked_probs = action_prob * state_reshaped

            # Add small epsilon to prevent zero probabilities
            epsilon = 1e-10
            masked_probs = masked_probs + epsilon

            # Normalize probabilities
            normalized_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)

            # Try to find valid word
            for _ in range(10):
                try:
                    # Sample letters from valid probabilities
                    sampled_letters = torch.multinomial(action_prob, num_samples=1).squeeze()
                    word = ''.join([chr(ord('a') + idx.item()) for idx in sampled_letters])

                    # Check if word is valid
                    if word in self.word_to_idx:
                        action = self.word_to_idx[word]
                        # Calculate probability correctly (product of chosen letters)
                        letter_probs = action_prob[torch.arange(5), sampled_letters]
                        old_action_prob_selected = letter_probs.prod()
                        return action, old_action_prob_selected

                except RuntimeError:
                    # If sampling fails, use argmax instead
                    sampled_letters = torch.argmax(normalized_probs, dim=1)
                    continue

            # Fallback to random word
            action = random.randrange(len(self.env.allowed_words))
            return action, torch.tensor(1e-8, device=device)

    def batch_update(self, states, actions, rewards, next_states, old_action_probs_selected, dones):
        states = torch.stack(states)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.stack(next_states)
        old_action_probs_selected = torch.tensor(old_action_probs_selected, device=device, dtype=torch.float32)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)

        """# Print batch information with rewards
        print("\nBatch details:")
        print(f"Batch size: {len(actions)}")
        for i, (action_idx, reward, done) in enumerate(zip(actions, rewards, dones)):
            guess_word = self.env.allowed_words[action_idx]
            result = "Won" if done and reward > 0 else "Lost" if done else "In progress"
            print(f"{i + 1}. Word: {guess_word}, Reward: {reward:.2f}, Status: {result}")
        print("-" * 50)"""

        # Critic update
        with torch.no_grad():
            current_values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
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

        for _ in range(self.actor_repetition):
            self.optimizer_actor.zero_grad()

            # Get new action probabilities
            # Get actor prob (no softmax yet)
            prob = self.actor(states)  # Shape: [batch_size, 5*26]
            prob = prob.view(-1, self.env.word_length, 26)  # Reshape to [batch_size, 5, 26]

            # Apply masking to prob (states_reshaped is 1 for valid letters)
            states_reshaped = states.view(-1, self.env.word_length, 26)
            masked_prob = prob + (states_reshaped - 1) * 1e10  # Mask invalid letters

            # Apply softmax per position
            new_action_probs = torch.softmax(masked_prob, dim=2)  # Shape: [batch_size, 5, 26] (softmax over all 5 positions)
            states_reshaped = states.view(-1, self.env.word_length, 26)  # Shape: [batch_size, 5, 26]

            # Mask and normalize probabilities
            masked_probs = new_action_probs * states_reshaped
            normalized_probs = masked_probs / (masked_probs.sum(dim=2, keepdim=True) + 1e-10)

            # Get probabilities for selected actions
            word_probs = []
            for i, action_idx in enumerate(actions):
                word = self.env.allowed_words[action_idx]
                letter_indices = torch.tensor([[ord(c) - ord('a') for c in word]], device=device)
                pos_indices = torch.arange(self.env.word_length, device=device).unsqueeze(0)
                word_prob = normalized_probs[i, pos_indices, letter_indices].prod()
                word_probs.append(word_prob)
            new_action_probs_selected = torch.stack(word_probs)

            # Compute PPO loss
            importance_ratios = new_action_probs_selected / (old_action_probs_selected + 1e-10)
            clipped_ratios = torch.clamp(importance_ratios, 1 - self.epsilon, 1 + self.epsilon)
            loss = torch.min(
                importance_ratios * td_errors_detached,
                clipped_ratios * td_errors_detached
            )
            actor_loss = -loss.mean()

            # Backward pass and update
            actor_loss.backward()
            self.optimizer_actor.step()
            actor_losses.append(actor_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    def train(self, epochs=500, print_freq=50):
        print("Training...")
        total_wins = 0
        buffer = []
        batch_losses_actor = []
        batch_losses_critic = []

        # Create or clear the metrics file at the start of training
        with open('training_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate'])

        for episode in tqdm(range(epochs)):
            self.env.reset()
            state = self.state()
            episode_transitions = []
            episode_reward = 0
            last_correct = 0
            last_in_word = 0

            # Play exactly 6 rounds or until game ends
            for round in range(self.env.max_tries):
                action, old_prob = self.act_word()
                matches = self.env.guess(self.env.allowed_words[action])
                next_state = self.state()
                done = self.env.end

                # Calculate improved reward
                correct_position = self.env.correct_position
                in_word = self.env.in_word

                # Reward for improvement
                position_improvement = correct_position - last_correct
                word_improvement = in_word - last_in_word

                # Stronger win rewards and clearer penalties
                if self.env.win:
                    reward = 10.0 + (self.env.max_tries - round) * 1.0  # Much higher win reward
                else:

                    position_reward = position_improvement * 0.5
                    word_reward = word_improvement * 0.3

                    base_penalty = -0.1 if position_improvement == 0 and word_improvement == 0 else -0.05

                    reward = position_reward + word_reward + base_penalty

                    if done and not self.env.win:
                        reward -= 1.0

                # Update last state
                last_correct = correct_position
                last_in_word = in_word

                episode_reward += reward
                episode_transitions.append((state, action, reward, next_state, old_prob, done))

                if done:
                    break

                state = next_state.clone()

            # Update stats
            total_wins += self.env.win
            self.stats['wins'] += self.env.win
            self.stats['total_games'] += 1
            self.stats['tries_distribution'][self.env.try_count] += 1
            self.stats['results'][self.env.word] = {'tries': self.env.try_count, 'win': self.env.win}

            # Add transitions to buffer
            buffer.extend(episode_transitions)

            # Perform batch update when buffer is large enough
            if len(buffer) >= self.batch_size:
                # Process as many complete batches as possible
                num_complete_batches = len(buffer) // self.batch_size
                for _ in range(num_complete_batches):
                    batch = buffer[:self.batch_size]
                    buffer = buffer[self.batch_size:]
                    states, actions, rewards, next_states, old_probs, dones = zip(*batch)
                    loss_actor, loss_critic = self.batch_update(states, actions, rewards, next_states, old_probs, dones)
                    batch_losses_actor.append(loss_actor)
                    batch_losses_critic.append(loss_critic)

            # Print stats + save model + save metrics
            if (episode + 1) % print_freq == 0:
                avg_loss_actor = np.mean(batch_losses_actor) if batch_losses_actor else 0
                avg_loss_critic = np.mean(batch_losses_critic) if batch_losses_critic else 0
                win_rate = total_wins / print_freq

                # Save metrics to file
                self.save_training_metrics(episode + 1, avg_loss_actor, avg_loss_critic, win_rate)

                print(f"Episode {episode + 1}/{epochs} - Actor Loss: {avg_loss_actor:.4f}, "
                      f"Critic Loss: {avg_loss_critic:.4f}, Win Rate: {win_rate:.4f}")
                total_wins = 0
                batch_losses_actor = []
                batch_losses_critic = []
                self.save_model(f'actor_critic_{episode + 1}.pt')
                self.save_stats('actor_critic_stats.pkl')

        # Process remaining buffer data
        if len(buffer) >= self.batch_size:
            states, actions, rewards, next_states, old_probs, dones = zip(*buffer[:self.batch_size])
            loss_actor, loss_critic = self.batch_update(states, actions, rewards, next_states, old_probs, dones)
            print(f"Final batch - Actor Loss: {loss_actor:.4f}, Critic Loss: {loss_critic:.4f}")

        self.save_model('actor_critic_end.pt')
        self.save_stats('actor_critic_stats.pkl')
        print("Training finished.")

    def continue_training(self, model_path, stats_path=None, epochs=5000, print_freq=500,
                          learning_rate=None, epsilon=None, actor_repetition=None, critic_repetition=None):

        # Load the saved model
        self.load_model(model_path)
        print(f"Loaded model from {model_path}")

        # Load stats if provided
        if stats_path and os.path.exists(stats_path):
            self.load_stats(stats_path)
            win_rate = self.stats['wins'] / self.stats['total_games'] if self.stats['total_games'] > 0 else 0
            print(f"Loaded stats from {stats_path}: {self.stats['total_games']} games, win rate: {win_rate:.4f}")

        # Update hyperparameters if specified
        if learning_rate is not None:
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
            print(f"Updated learning rate to {learning_rate}")

        if epsilon is not None:
            self.epsilon = epsilon
            print(f"Updated epsilon to {epsilon}")

        if actor_repetition is not None:
            self.actor_repetition = actor_repetition
            print(f"Updated actor repetition to {actor_repetition}")

        if critic_repetition is not None:
            self.critic_repetition = critic_repetition
            print(f"Updated critic repetition to {critic_repetition}")

        # Continue training
        print(f"Continuing training for {epochs} epochs...")
        self.train(epochs=epochs, print_freq=print_freq)

    def run_test(self, path, num_games):
        self.load_model(path)
        total_wins = 0
        total_tries = 0
        for _ in tqdm(range(num_games)):
            self.env.reset()
            while not self.env.end:
                state = self.state()
                action, _ = self.act_individual_letter()
                self.env.guess(self.env.allowed_words[action])
            total_tries += self.env.try_count
            if self.env.win:
                total_wins += 1
        avg_tries = total_tries / num_games
        win_rate = total_wins / num_games
        print(f"Average tries: {avg_tries}, Win rate: {win_rate}")
        return avg_tries, win_rate


# Example usage
env = Environment('wordle-nyt-allowed-guesses-update-12546.txt')
A = Actor(env, epsilon=0.1, learning_rate=1e-4, actor_repetition=10, critic_repetition=2)
A.continue_training(model_path='actor_critic_10000_word_from_letter.pt', stats_path='actor_critic_stats.pkl', epochs=10000, print_freq=500,)
