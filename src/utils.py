import matplotlib.pyplot as plt


def get_action_name(idx):
    action_dict = {
        0: "forward",
        1: "backward",
        2: "left",
        3: "right"
    }
    return action_dict[idx]


def plot_rewards(rewards):
    plt.title("Summary of rewards")
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.show()

