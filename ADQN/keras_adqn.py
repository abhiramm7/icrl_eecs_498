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
        model.add(BatchNormalization())
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    sgd = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

q_nn = build_network(4, 6, 1, 32, "relu", 0.0);
target_nn = build_network(4, 6, 1, 32, "relu", 0.0);


target_nn.set_weights(q_nn.get_weights())

replay1 = replay_memory_agent(4, 10000)

dqn_controller = deep_q_agent(action_value_model=q_nn,
                              target_model=target_nn,
                              states_len=4,
                              replay_memory=replay1)

env = gym.make("CartPole-v0")

check_save = -100
# Book keeping
avg_reward_episodes = []
# Global time step
gt = 0
epsilon=1.0
for episodes in range(0, 8000):

    # Initial State
    state = env.reset()
    done_1=False
    done_2=False
    
    # Clear the reward buffer
    rewards = []
    if gt > 1000:
        epsilon = max(0.10, epsilon-0.0009)
    else:
        epsilon = 1.0
    
    episode_time = 0

    while not(done_1) and not(done_2):
        gt += 1

        # Reshape the state
        state = np.asarray(state)
        state = state.reshape(1,4)

        # Pick a action based on the state
        q_values = q_nn.predict_on_batch(state)

        if np.random.rand() <= epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_values)

        if action == 2:
            state_1, reward_1, done_1, _ = env.step(0)
            action_1 = 0

            rewards.append(reward_1)

            if not done_1:

                state_2, reward_2, done_2, _ = env.step(0)
                action_2 = 0

                replay1.replay_memory_update(state,
                        state_1,
                        state_2,
                        reward_1,
                        reward_2,
                        action_1,
                        action_2,
                        done_1,
                        done_2)

                state = state_2 

            else:
                replay1.replay_memory_update(state,
                        state_1,
                        [0.0, 0.0, 0.0, 0.0],
                        reward_1,
                        0.0,
                        action_1,
                        0.0,
                        done_1,
                        False)
                break

        elif action == 3:

            state_1, reward_1, done_1, _ = env.step(0)
            action_1 = 0

            rewards.append(reward_1)

            if not done_1:

                state_2, reward_2, done_2, _ = env.step(1)
                action_2 = 1

                replay1.replay_memory_update(state,
                        state_1,
                        state_2,
                        reward_1,
                        reward_2,
                        action_1,
                        action_2,
                        done_1,
                        done_2)

                state = state_2

            else:
                replay1.replay_memory_update(state,
                        state_1,
                        [0.0, 0.0, 0.0, 0.0],
                        reward_1,
                        0.0,
                        action_1,
                        0.0,
                        done_1,
                        False)
                break

        elif action == 4:
            state_1, reward_1, done_1, _ = env.step(1)
            action_1 = 1

            rewards.append(reward_1)

            if not done_1:

                state_2, reward_2, done_2, _ = env.step(0)
                action_2 = 0

                replay1.replay_memory_update(state,
                        state_1,
                        state_2,
                        reward_1,
                        reward_2,
                        action_1,
                        action_2,
                        done_1,
                        done_2)

                state = state_2 

            else:
                replay1.replay_memory_update(state,
                        state_1,
                        [0.0, 0.0, 0.0, 0.0],
                        reward_1,
                        0.0,
                        action_1,
                        0.0,
                        done_1,
                        False)
                break

        elif action == 5:

            state_1, reward_1, done_1, _ = env.step(1)
            action_1 = 1

            rewards.append(reward_1)

            if not done_1:
                state_2, reward_2, done_2, _ = env.step(1)
                action_2 = 1

                replay1.replay_memory_update(state,
                        state_1,
                        state_2,
                        reward_1,
                        reward_2,
                        action_1,
                        action_2,
                        done_1,
                        done_2)

                state = state_2 

            else:
                replay1.replay_memory_update(state,
                        state_1,
                        [0.0, 0.0, 0.0, 0.0],
                        reward_1,
                        0.0,
                        action_1,
                        0.0,
                        done_1,
                        False)
                break

        if action == 0 or action == 1:

            # Implement action and observe the reward signal
            state_1, reward_1, done_1, _ = env.step(action)
            rewards.append(reward_1)

            # Update the replay memory
            replay1.replay_memory_update(state,
                    state_1,
                    [0.0, 0.0, 0.0, 0.0],
                    reward_1,
                    0.0,
                    action,
                    0.0,
                    done_1,
                    False)

            state = state_1


        if gt >= 1000:
            # Train
            update = True if gt%500==0 else False
            dqn_controller.train_q(update)
            if update:
                print("updated ", gt)


        episode_time += 1
        if episode_time > 250:
            break

    avg_reward_episodes.append(sum(rewards))
    if episodes%10 == 0 :
        print(sum(rewards), "Episode Count :" ,episodes)
        q_nn.save_weights("model"+str(episodes))

np.save("sum_rewards", avg_reward_episodes)
plt.plot(avg_reward_episodes)
plt.show()
