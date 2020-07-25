import matplotlib.pyplot as plt


def plot_log(frame_idx, rewards):
    plt.figure(figsize=(20, 20))
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    test_rewards = [1, 2, 300]
    plot_log(3, test_rewards)
