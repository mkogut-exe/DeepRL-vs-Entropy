import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Wordle import Environment
import random
from tqdm import tqdm
import pickle

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



class Actor:
    def __init__(self, env: Environment, batch_size=256, discount=0.99,epsilon=0.1,greedy=0.1 , learning_rate=1e-4,
                 actor_repetition=15, critic_repetition=5):
        self.env = env#initialize the environment
        self.discount = discount#discount factor
        self.batch_size = batch_size#batch size
        self.actor_repetition = actor_repetition#number of times the actor network is updated
        self.critic_repetition = critic_repetition#number of times the critic network is updated
        self.epsilon = epsilon#epsilon for the PPO algorithm
        self.greedy = greedy#epsilon greedy factor for exploration

        self.allowed_words_length = len(self.env.allowed_words)#length of the allowed words list
        self.word_to_idx = {word: idx for idx, word in enumerate(self.env.allowed_words)}#dictionary to convert words to indices
        self.allowed_words_tensor = torch.tensor([self.word_to_idx[w] for w in self.env.allowed_words], device=device)#convert the allowed words to a tensor of indices (for faster computation)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.allowed_words_length, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128,self.allowed_words_length ),
        nn.Softmax(dim=-1)
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.allowed_words_length, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(device)

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.stats = {
            'total_games': 0,
            'wins': 0,
            'win_rate': 0,
            'tries_distribution': {i: 0 for i in range(0, 8)},  # Include 0
            'results': {}
        }


    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        return self.stats



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

    def act(self):
        with torch.no_grad():
            state = self.state()
            action_prob = self.actor(state)
            action_prob = action_prob * state
            action_prob = action_prob / torch.sum(action_prob)
            action = torch.argmax(action_prob).item()  # pick action with max probability

            if random.random() < self.greedy:
                allowed_actions = torch.nonzero(state).view(-1)
                action = allowed_actions[random.randint(0, len(allowed_actions) - 1)].item()
            else:
                action = torch.argmax(action_prob).item()

        return action, action_prob

    def many_rewards_train(self, epochs=500, print_freq=50):
        print("Training...")
        total_wins=0
        wins_in_period=0
        step=0
        for epoch in tqdm(range(epochs)):
            step = 1  # Reset step counter for each episode
            self.env.reset(word_test=None)

            state = self.state()
            total_reward = torch.zeros(1, device=device)
            actions = []
            current_value = torch.zeros(1, device=device)

            while not self.env.end:
                old_action, old_action_prob = self.act()
                immediate_reward = torch.zeros(1, device=device)
                if epoch > print_freq and epoch % print_freq == 1:
                    print(f"\nAction: {self.env.allowed_words[old_action]}, Action prob: {old_action_prob[old_action]:.6f}")
                    if old_action in actions:
                            immediate_reward-=10
                            print("Repeated action!")

                self.env.guess(self.env.allowed_words[old_action])


                if self.env.win:
                    immediate_reward = torch.tensor([10.0 - (0.5/step)], device=device)
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
                        if old_action in actions:

                            print("Repeated action!")




                self.env.guess(self.env.allowed_words[old_action])

                if self.env.win:
                    total_reward+= 1.0

                total_reward-=0.1
                

                actions.append(old_action)


                next_state = self.state()
                action, action_prob, critic_reward, loss_actor, loss_critic = self.many_update(state, total_reward, next_state,
                                                                                   old_action_prob,
                                                                                   game_over=self.env.end)

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




A=Actor(Environment('wordle-nyt-allowed-guesses-update-12546.txt'),epsilon=0.1,greedy=0.01 , learning_rate=3e-5,actor_repetition=10, critic_repetition=1)
A.few_rewards_train(10000, print_freq=500)

