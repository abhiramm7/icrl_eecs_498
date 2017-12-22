from src import replay_memory_agent, deep_q_agent, epsi_greedy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import gym

def build_network(input_states,
                  output_states,
                  hidden_layers,
                  nuron_count,
                  activation_function,
                  dropout):
    """
    Build and initialize the neural network with a choice for dropout
    """
    model = Sequential()
    model.add(Dense(nuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for i_layers in range(0, hidden_layers - 1):
        model.add(Dense(nuron_count))     
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model



q_nn = build_network(4, 6, 1, 32, "relu", 0.0);
env = gym.make("CartPole-v0")

# Book keeping
model_reward = []
act = [0, 1, 2, 3, 4, 5]
time_act = {}
for i in range(0,9):
    time_act[i] = [0, 0, 0, 0, 0, 0]

# Global time step
gt = 0

for weights in np.linspace(0, 8000, 801, dtype=int):

    q_nn.load_weights("model"+str(weights))

    avg_reward_episodes = []
    
    inds = int(weights/1000)

    for episode_count in range(0, 1):
        # Initial State
        state = env.reset()
        done=False
            
        episode_time = 0
        
        rewards = []
        while not(done):
            gt += 1

            # Reshape the state
            state = np.asarray(state)
            state = state.reshape(1,4)

            # Pick a action based on the state
            q_values = q_nn.predict_on_batch(state)

            action = np.argmax(q_values)
            
            time_act[inds][action] += 1 

            if action <= 1:
            # Implement action and observe the reward signal
                state_new, reward, done, _ = env.step(action)
                rewards.append(reward)
            elif action == 2:
                state_new, reward, done, _ = env.step(0)
                rewards.append(reward)
                if done:
                    break
                state_new, reward, done, _ = env.step(0)
                rewards.append(reward)
            elif action == 3:
                state_new, reward, done, _ = env.step(0)
                rewards.append(reward)
                if done:
                    break
                state_new, reward, done, _ = env.step(1)
                rewards.append(reward)
            elif action == 4:
                state_new, reward, done, _ = env.step(1)
                rewards.append(reward)
                if done:
                    break
                state_new, reward, done, _ = env.step(0)
                rewards.append(reward)
            elif action == 5:
                state_new, reward, done, _ = env.step(1)
                rewards.append(reward)
                if done:
                    break
                state_new, reward, done, _ = env.step(1)
                rewards.append(reward)

            state = state_new

            episode_time += 1
            if episode_time > 200:
                break

        avg_reward_episodes.append(sum(rewards))
    
    model_reward.append(np.mean(avg_reward_episodes))

np.save("sum_rewards_adqn_with_target", model_reward)
np.save("action_space_adqn_with_target", time_act)
plt.plot(model_reward)
plt.show()

