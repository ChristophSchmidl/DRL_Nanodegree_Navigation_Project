from unityagents import UnityEnvironment
import torch
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import time
import src.agents as Agents


def print_device_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Pytorch version {torch.__version__} on device {device}")
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def print_env_info(env):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print(f"brain_name: {brain_name}")
    print(f"brain: {brain}")

    # reset the environment
    banana_env = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = banana_env.vector_observations[0]
    state_size = len(state)

    print(f"action_size: {action_size}")
    print(f"state_size: {state_size}")
    print(f"First observation/state: {state}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deep Q-Learning - Navigation Project'
    )

    # the hyphen makes the argument optional
    parser.add_argument('-episodes', type=int, default=1000, help='Number of games/episodes to play. Default is 1000.')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate for the optimizer. Default is 0.0001.')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation')

    parser.add_argument('-epsilon_start', type=float, default=1.0, help='Starting value for epsilon. Default is 1.0.')
    parser.add_argument('-epsilon_min', type=float, default=0.01, help='Minimum value for epsilon in epsilon-greedy action selection. Default is 0.01.')
    parser.add_argument('-epsilon_dec', type=float, default=1e-5, help='Rate of decay for epsilon. Default is 1e-5.')
    
    parser.add_argument('-buffer_size', type=int, default=30000, help='Maximum size of memory/replay buffer. Default is 30000.')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training. Default is 32.')
    parser.add_argument('-update_target', type=int, default=1000, help='Interval (of steps) for updating/replacing target network. Default is 1000.')

    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1. Default is 0.')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='Load model checkpoint/weights. Default is False.')
    parser.add_argument('-model_path', type=str, default='data/',
                        help='Path for model saving/loading. Default is data/')
    parser.add_argument('-plot_path', type=str, default='plots/',
                        help='Path for saving plots. Default is plots/')
    parser.add_argument('-save_plot', type=bool, default=True,
                        help='Save plot of eval or/and training phase. Default is True.')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                    help='You can use the following algorithms: DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent. Default is DQNAgent.')
    parser.add_argument('-eval', type=bool, default=False,
                        help='Evaluate the agent. Deterministic behavior. Default is False.')
    parser.add_argument('-visual_env', type=bool, default=False,
                        help='Using the visual environment. Default is False.')
    args = parser.parse_args()

    # set GPU (if you have multiple GPUs)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print_device_info()

    if args.visual_env:
        env = UnityEnvironment(file_name="src/Banana_Linux/Banana.x86_64")
    else:
        env = UnityEnvironment(file_name="src/Banana_Linux_NoVis/Banana.x86_64")
    
    print_env_info(env)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    banana_env = env.reset(train_mode=True)[brain_name] # True means that the environment speed is faster (use False for visualiation)
    action_size = brain.vector_action_space_size
    state = banana_env.vector_observations[0]
    state_size = len(state)

    ##########################################
    #       Training/Evaluation loop
    ##########################################
    best_score = -np.inf
    n_steps = 0
    episode_rewards, epsilon_history, steps_array = [], [], []

    # Neat trick without using if/else/switch: get me the correct agent/algorithm
    conrete_agent = getattr(Agents, args.algo)

    agent = conrete_agent(gamma=args.gamma,
                  epsilon=args.epsilon_start,
                  lr=args.lr,
                  #input_dims=env.observation_space.shape,
                  input_dims=state_size,
                  #n_actions=env.action_space.n,
                  n_actions=action_size,
                  buffer_size=args.buffer_size,
                  epsilon_min=args.epsilon_min,
                  batch_size=args.batch_size,
                  update_target=args.update_target,
                  epsilon_dec=args.epsilon_dec,
                  checkpoint_dir=args.model_path,
                  algo=args.algo,
                  #env_name=args.env
                  env_name='Banana'
    )

    # If in evaluation mode
    if args.eval:
        print("Evaluating agent...")
        # This leads to a deterministic behavior without exploration
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0
        agent.epsilon_dec = 0.0
        args.load_checkpoint = True

    if args.load_checkpoint:
        agent.load_models()

    filename = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + \
                str(args.episodes) + 'episodes'
    figure_file = args.plot_path + filename + '.png'
    
    start_time = time.time()
    solution_txt =""

    for i in range(args.episodes):
        done = False
        score = 0

        if args.eval and args.visual_env:
            banana_env = env.reset(train_mode=False)[brain_name] # Slower environment for visualization
        else:
            banana_env = env.reset(train_mode=True)[brain_name]

        obs = banana_env.vector_observations[0]

        while not done:
            action = agent.choose_action(obs)
            banana_env = env.step(action)[brain_name] 
            next_obs = banana_env.vector_observations[0]   # get the next state
            reward = banana_env.rewards[0]                   # get the reward
            done = banana_env.local_done[0]  

            score += reward
        
            if not args.load_checkpoint:
                agent.store_transition(obs, action, reward, next_obs, int(done))
                agent.learn()
            
            obs = next_obs
            n_steps += 1

        episode_rewards.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(episode_rewards[-100:])
        print(f"Episode: {i}, Score: {score:.2f}, Average score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {n_steps}")

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        epsilon_history.append(agent.epsilon)
        
        if avg_score >= 13 and not args.eval:
            solution_txt = f"Solved in {i} episodes with an average reward score of {avg_score:.2f} of the last 100 episodes"
            print(solution_txt)
            break

    end_time = (time.time() - start_time)/60
    print(f"\nTotal training time = {end_time:.1f} minutes")

    # plot the scores
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111)
    #ax = fig.add_subplot(111)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)

    if args.eval:
        plt.title(f"Navigation project: {args.algo} - Evaluation") 
    else:
        plt.title(f"Navigation project: {args.algo} - Training") 

    plt.ylabel('Rewards')
    new_line = '\n'
    plt.xlabel(f"Episodes{new_line}{solution_txt}")
    plt.grid(True)

    if args.save_plot:
        if args.eval:
            plt.savefig(f"{args.plot_path}/Navigation_project_{args.algo}_eval.png")
        else:
            plt.savefig(f"{args.plot_path}/Navigation_project_{args.algo}_train.png")
    plt.show()
    
    env.close()