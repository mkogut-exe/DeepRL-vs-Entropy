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
"""
Wordle agent using deep reinforcement learning with input of word vector representation of the all matching words and outputs guessed word.

"""





#check torch versin and cuda availability
torch_version = torch.__version__
cuda_available = torch.cuda.is_available()

#turn on torch gpu mode if available
device = torch.device('cuda' if cuda_available else 'cpu')
print(f'Torch version: {torch_version}, CUDA availability: {cuda_available}, Device: {device}')


# Set the random seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



def create_model_id(epochs, actor_repetition, critic_repetition, actor_network_size,learning_rate,batch_size):
    return f"_WV_epo-{epochs}_AR-{actor_repetition}_CR-{critic_repetition}_AS-{actor_network_size}-Lr-{learning_rate}-Bs-{batch_size}"



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

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.allowed_words_length, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256,self.allowed_words_length ),
        nn.Softmax(dim=-1)
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.allowed_words_length, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        ).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

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

    def save_training_metrics(self, episode, actor_loss, critic_loss, win_rate, metrics_file='training_metrics'):
        full_path = f"{metrics_file}{self.model_id}.csv"
        file_exists = os.path.isfile(full_path)

        with open(full_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate'])
            writer.writerow([episode, actor_loss, critic_loss, win_rate])




    def state(self):
        matches = self.env.find_matches()
        # Convert matches to indices using a precomputed dictionary
        matches_indices = [self.word_to_idx[match] for match in matches]
        # Create state_vector directly on GPU
        state_vector = torch.zeros(self.allowed_words_length, device=device)#create a tensor of zeros with the length of the allowed words
        if matches_indices:#if there are matching words make them = 1 in the state vector
            indices_tensor = torch.tensor(matches_indices, device=device)
            state_vector[indices_tensor] = 1.0
        return state_vector
    def one_update(self, state, reward, next_state, old_action_prob,game_over):#update the actor and critic networks with Proximal Policy Optimization using TD(0)

        #############################################Critic update#############################################

        # Reset gradients before critic update
        self.optimizer_critic.zero_grad()

        critic_value = self.critic(state)#feed the current state to the critic network to get the value
        next_critic_value = self.critic(next_state).detach()#feed the next state to the critic network to get the value (but don't compute the gradient)
        target_value = reward + self.discount * next_critic_value * (1 - game_over)#compute the target value (1-game_over is 0 if the game is over and 1 otherwise)
        td_error = (target_value - critic_value).detach()  # Calculate TD error for actor

        loss_critic = nn.MSELoss()(critic_value, target_value)#compute the Mean Squered Error loss for the critic

        # Compute the gradients
        loss_critic.backward(retain_graph=True)

        # Update the parameters
        self.optimizer_critic.step()


        #############################################Actor update#############################################

        # Reset gradients before actor update
        self.optimizer_actor.zero_grad()

        action_prob = self.actor(state)#feed the current state to the actor network to get the action probabilities
        action_prob = action_prob*state#mask the action probabilities to filter out not matching options
        action_prob = action_prob/torch.sum(action_prob)#normalize the action probabilities
        importance_sampling = action_prob/(old_action_prob + 1e-10)#compute importance sampling ratio
        loss_actor = torch.min(td_error*importance_sampling, td_error*torch.clamp(importance_sampling, 1-self.epsilon, 1+self.epsilon))#compute the loss for the actor
        loss_actor = -torch.mean(loss_actor)#compute the mean loss ('-' because we want to maximize the reward)

        # Compute gradients
        loss_actor.backward()

        # Update parameters
        self.optimizer_actor.step()


        action = torch.argmax(action_prob).item() #pick action with max probability

        return action, action_prob, critic_value, loss_actor.item(), loss_critic.item()

    def many_update(self, state, reward, next_state, old_action_prob, game_over):
        #############################################Critic update#############################################
        # Store initial critic value for TD error
        with torch.no_grad():
            initial_critic_value = self.critic(state)
            next_critic_value = self.critic(next_state)
            target_value = reward + self.discount * next_critic_value * (1 - game_over)
            td_error = (target_value - initial_critic_value)

        # Perform critic updates
        for _ in range(self.critic_repetition):
            self.optimizer_critic.zero_grad()
            critic_value = self.critic(state)
            loss_critic = nn.MSELoss()(critic_value, target_value)
            loss_critic.backward(retain_graph=True)
            self.optimizer_critic.step()

        with torch.no_grad():
            initial_critic_value = self.critic(state)
            next_critic_value = self.critic(next_state)
            target_value = reward + self.discount * next_critic_value * (1 - game_over)
            td_error = (target_value - initial_critic_value)

        #############################################Actor update#############################################
        # Store initial probabilities
        with torch.no_grad():
            initial_action_prob = self.actor(state)
            initial_action_prob = initial_action_prob * state
            initial_action_prob = initial_action_prob / torch.sum(initial_action_prob)

        # Perform actor updates
        for _ in range(self.actor_repetition):
            self.optimizer_actor.zero_grad()
            action_prob = self.actor(state)
            action_prob = action_prob * state
            action_prob = action_prob / torch.sum(action_prob)
            importance_sampling = action_prob / (old_action_prob + 1e-10)
            loss_actor = torch.min(
                td_error * importance_sampling,
                td_error * torch.clamp(importance_sampling, 1 - self.epsilon, 1 + self.epsilon)
            )
            loss_actor = -torch.mean(loss_actor)
            loss_actor.backward(retain_graph=True)
            self.optimizer_actor.step()

        # Get final action
        with torch.no_grad():
            final_action_prob = self.actor(state)
            final_action_prob = final_action_prob * state
            final_action_prob = final_action_prob / torch.sum(final_action_prob)
            action = torch.argmax(final_action_prob).item()

        return action, final_action_prob, critic_value, loss_actor.item(), loss_critic.item()

    def batch_update(self, states, actions, rewards, next_states, old_action_probs_selected, dones):
        states = torch.stack(states)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.stack(next_states)
        old_action_probs_selected = torch.tensor(old_action_probs_selected, device=device, dtype=torch.float32)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)

        # Critic update
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

            # If random sampling is enabled, select a subset
            if self.random_batch and batch_size > self.sample_size:
                indices = torch.randperm(batch_size, device=device)[:self.sample_size]
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_td_errors = td_errors_detached[indices]
                batch_old_probs = old_action_probs_selected[indices]
            else:
                batch_states = states
                batch_actions = actions
                batch_td_errors = td_errors_detached
                batch_old_probs = old_action_probs_selected

            # Get new action probabilities for word vectors
            new_action_probs = self.actor(batch_states)

            # Apply mask for valid words
            masked_probs = new_action_probs * batch_states

            # Normalize probabilities
            sums = masked_probs.sum(dim=1, keepdim=True)
            normalized_probs = masked_probs / (sums + 1e-10)

            # Get probabilities for selected actions
            batch_indices = torch.arange(len(batch_actions), device=device)
            new_action_probs_selected = normalized_probs[batch_indices, batch_actions]

            # Compute PPO loss
            importance_ratios = new_action_probs_selected / (batch_old_probs + 1e-10)
            clipped_ratios = torch.clamp(importance_ratios, 1 - self.epsilon, 1 + self.epsilon)
            loss = torch.min(
                importance_ratios * batch_td_errors,
                clipped_ratios * batch_td_errors
            )
            actor_loss = -loss.mean()

            # Backward pass and update
            actor_loss.backward()
            self.optimizer_actor.step()
            actor_losses.append(actor_loss.item())

        # Return the mean losses
        return np.mean(actor_losses), np.mean(critic_losses)

    def train(self, epochs=500, print_freq=50, autosave=False, append_metrics=False, prune_amount=0.1, prune_freq=1000,
              sparsity_threshold=0.1, prune=False):
        print("Training...")
        self.prune_amount = prune_amount
        self.prune_freq = prune_freq
        self.sparsity_threshold = sparsity_threshold
        self.prune = prune
        self.model_id = create_model_id(epochs=epochs, actor_repetition=self.actor_repetition,
                                        critic_repetition=self.critic_repetition, actor_network_size='1x256',
                                        learning_rate=self.learning_rate, batch_size=self.batch_size)
        total_wins = 0
        batch_losses_actor = []
        batch_losses_critic = []

        # Initialize replay buffer
        replay_buffer = []

        # Create or append to metrics file
        with open(f'training_metrics{self.model_id}.csv', 'a' if append_metrics else 'w', newline='') as f:
            writer = csv.writer(f)
            if not append_metrics:
                writer.writerow(['Episode', 'Actor_Loss', 'Critic_Loss', 'Win_Rate'])

        for episode in tqdm(range(epochs)):
            self.env.reset()
            state = self.state()
            last_correct = 0
            last_in_word = 0

            for round in range(self.env.max_tries):
                action, old_prob = self.act()
                matches = self.env.guess(self.env.allowed_words[action])
                next_state = self.state()
                done = self.env.end

                # Calculate reward
                correct_position = self.env.correct_position
                in_word = self.env.in_word
                position_improvement = correct_position - last_correct
                word_improvement = in_word - last_in_word

                if self.env.win:
                    # Keep exponential bonus for faster wins
                    time_bonus = 0.4 ** (self.env.try_count - 1)
                    reward = 2.0 + time_bonus  # Base win reward + scaled bonus
                else:

                    # Make improvements non-negative and cap them
                    position_improvement = max(0, position_improvement)
                    word_improvement = max(0, word_improvement)

                    # Apply diminishing returns to limit small improvement rewards
                    position_reward = min(0.5, position_improvement * 0.3)
                    word_reward = min(0.3, word_improvement * 0.2)

                    # Progress bonus (small but positive)
                    progress_bonus = 0.05 if (position_improvement > 0 or word_improvement > 0) else 0

                    # Final positive reward
                    reward = position_reward + word_reward + progress_bonus

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

                self.save_training_metrics(episode + 1, avg_loss_actor, avg_loss_critic, win_rate)

                print(f"Episode {episode + 1}/{epochs} - Actor Loss: {avg_loss_actor:.4f}, "
                      f"Critic Loss: {avg_loss_critic:.4f}, Win Rate: {win_rate:.4f}")

                total_wins = 0
                batch_losses_actor = []
                batch_losses_critic = []

                if autosave:
                    self.save_model(f'actor_critic_{episode + 1}{self.model_id}.pt')
                self.save_stats(f'actor_critic_stats{self.model_id}.pkl')

        # Process any remaining samples in the buffer at the end
        while len(replay_buffer) >= 32:  # Process remaining data in small batches
            mini_batch_size = min(1024, len(replay_buffer))
            batch = replay_buffer[:mini_batch_size]
            replay_buffer = replay_buffer[mini_batch_size:]

            states, actions, rewards, next_states, old_probs, dones = zip(*batch)
            self.batch_update(states, actions, rewards, next_states, old_probs, dones)

        self.save_model(f'actor_critic_end{self.model_id}.pt')
        self.save_stats(f'actor_critic_stats{self.model_id}.pkl')
        print("Training finished.")

    def act(self):
        with torch.no_grad():
            state = self.state()
            action_prob = self.actor(state)
            action_prob = action_prob * state

            # Normalize probabilities
            total_prob = torch.sum(action_prob)
            if total_prob > 0:
                action_prob = action_prob / total_prob

            action = torch.argmax(action_prob).item()

            # Return action and the scalar probability of that action
            return action, action_prob[action].item()

    def many_rewards_train(self, epochs=500, print_freq=50):
        print("Training...")
        total_wins=0
        wins_in_period=0
        step=0
        for epoch in tqdm(range(epochs)):
            step = 1  # Reset step counter for each episode
            self.env.reset(word_test=None)


            total_reward = torch.zeros(1, device=device)
            actions = []

            current_value = torch.zeros(1, device=device)
            state = self.state()


            while not self.env.end:

                old_action, old_action_prob = self.act()
                immediate_reward = torch.zeros(1, device=device)
                if old_action in actions:
                    immediate_reward -= 1

                if epoch > print_freq and epoch % print_freq == 1:
                    print(f"\nAction: {self.env.allowed_words[old_action]}, Action prob: {old_action_prob[old_action]:.6f}")

                self.env.guess(self.env.allowed_words[old_action])


                if self.env.win:
                    immediate_reward = torch.tensor([1.0 - (0.1/step)], device=device)
                else:
                    # Reward for correct letters in position
                    immediate_reward = torch.tensor([
                        0.1 * self.env.correct_position +
                        0.05 * self.env.in_word -
                        0.1 * step  # Penalty for each step
                    ], device=device)

                total_reward += immediate_reward
                actions.append(old_action)
                next_state = self.state()

                action, action_prob, critic_value, loss_actor, loss_critic = self.many_update(
                    state,
                    immediate_reward,
                    next_state,
                    old_action_prob,
                    game_over=self.env.end
                )

                state = next_state
                step += 1

            wins_in_period += self.env.win
            total_wins += self.env.win

            if epoch > print_freq and epoch % print_freq == 1:
                self.save_model(f'actor_critic_{epoch}.pt')
                print(f'\nWord being guessed: {self.env.word}, Win: {self.env.win}, '
                      f'Total reward: {total_reward.item():.4f}, '
                      f'Critic value: {critic_value.item():.4f}, '
                      f'Total wins: {total_wins}, '
                      f'Win rate in period: {wins_in_period/print_freq:.4f}')
                print(f"{epoch}/{epochs} - Loss actor: {loss_actor:.4f}, Loss critic: {loss_critic:.4f}")
                wins_in_period = 0
        print("Training finished.Average win rate: ", total_wins / epochs)
        self.save_model(f'actor_critic_end.pt')
    def few_rewards_train(self, epochs=500, print_freq=50):
        print("Training...")
        total_wins=0
        wins_in_period=0
        for epoch in tqdm(range(epochs)):
            self.env.reset(word_test=None)
            state = self.state()

            total_reward = 0
            actions = []
            critic_reward=0
            while not self.env.end:

                old_action, old_action_prob = self.act()


                if epoch > print_freq:
                    if epoch % print_freq == 1:

                        print(
                            f"\nAction: {self.env.allowed_words[old_action]}, Action prob: {old_action_prob[old_action]:.6f}")





                self.env.guess(self.env.allowed_words[old_action])

                if self.env.win:
                    immediate_reward = 1.0  # Reward for winning
                else:
                    immediate_reward = -0.1  # Penalty per step

                if old_action in actions:
                    immediate_reward  -= 1

                actions.append(old_action)


                next_state = self.state()
                action, action_prob, critic_reward, loss_actor, loss_critic = self.many_update(state, immediate_reward , next_state,
                                                                                   old_action_prob,
                                                                                   game_over=self.env.end)
                state = next_state

            wins_in_period += self.env.win
            total_wins += self.env.win


            if epoch > print_freq:

                if epoch % print_freq == 1:
                    self.save_model(f'actor_critic_{epoch}.pt')
                    print(f'\nWord beeing guessed: {self.env.word}, Win: {self.env.win}, Total reward: {total_reward:.4f}, Critic reward: {critic_reward[0]:}, Total wins: {total_wins}, win rate in period: {wins_in_period/print_freq:.4f}')
                    print(f"{epoch}/{epochs} - Loss actor: {loss_actor:.4f}, Loss critic: {loss_critic:.4f}")
                    wins_in_period = 0
        print("Training finished.Average win rate: ", total_wins/epochs)
        self.save_model(f'actor_critic_end.pt')

    def run(self,state):
        action = 0
        action_prob = self.actor(state)
        action_prob = action_prob * state
        action_prob = action_prob / torch.sum(action_prob)
        action = torch.argmax(action_prob).item()
        return action
    def run_test(self,path,num_games):#run test on the model
        self.load_model(path)
        total_wins = 0
        total_tries = 0
        # Run test for num_games
        for _ in tqdm(range(num_games)):
            self.env.reset()
            while not self.env.end:
                state = self.state()
                action = self.run(state)
                self.env.guess(self.env.allowed_words[action])
            total_tries += self.env.try_count
            if self.env.win:
                total_wins += 1
                self.stats['wins'] += 1
            self.stats['total_games'] += 1
            self.stats['tries_distribution'][self.env.try_count] += 1
            self.stats['results'][self.env.word] = {'tries': self.env.try_count, 'win': self.env.win}
        avg_tries= total_tries / num_games
        avg_wins = total_wins / num_games
        self.stats['win_rate'] = avg_wins
        self.save_stats('actor_critic_stats.pkl')
        print(f"Average tries: {avg_tries}")
        print(f"Win rate: {avg_wins}")
        return avg_tries, avg_wins




env = Environment('thiny_set.txt')
A = Actor(env,batch_size=100, epsilon=0.1, learning_rate=1e-3, actor_repetition=10, critic_repetition=2,random_batch=True)
A.train(epochs=10000, print_freq=500,prune=False)