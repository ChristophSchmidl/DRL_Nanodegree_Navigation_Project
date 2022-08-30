import numpy as np
import numbers
import pickle
from replaybuffer import Experience


def is_number(v):
    return isinstance(v, numbers.Number)

def fill_memory(env, dqn_agent, num_memory_fill_eps):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print("*** Filling replaybuffer...***")
    for i in range(num_memory_fill_eps):
        print(i)
        done = False

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        while not done:
            #print("Entered while loop")
            action = np.random.randint(brain.vector_action_space_size)
            env_info = env.step(action)[brain_name]  # send the action to the environment

            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            #print(reward)
            done = env_info.local_done[0]  # see if episode has finished

            dqn_agent.memory.add_experience(Experience(state, action, reward, next_state, done))

    print("Saving experiences...")
    dqn_agent.memory.save_buffer()


def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batch_size, results_basepath,
          render=False):
    #fill_memory(env, dqn_agent, num_memory_fill_eps)
    #print('Memory filled. Current capacity: ', len(dqn_agent.memory))

    dqn_agent.memory.load_buffer()

    reward_history = []
    epsilon_history = []

    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.get_current_epsilon())

        done = False
        state = env.reset()

        ep_score = 0

        while not done:
            if render:
                env.render()

            action = dqn_agent.select_action(state)

            env_info = env.step(action)["BananaBrain"]  # send the action to the environment

            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            dqn_agent.memory.add_experience(Experience(state, action, reward, next_state, done))

            dqn_agent.learn(batch_size=batch_size)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:])  # moving average of last 100 episodes

        print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'.format(ep_cnt, step_cnt, ep_score,
                                                                                          current_avg_score,
                                                                                          epsilon_history[-1]))

        if current_avg_score >= best_score:
            dqn_agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score

    with open('{}/train_reward_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(epsilon_history, f)


def test(env, dqn_agent, num_test_eps, seed, results_basepath, render=True):
    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

    with open('{}/test_reward_history_{}.pkl'.format(results_basepath, seed), 'wb') as f:
        pickle.dump(reward_history, f)
