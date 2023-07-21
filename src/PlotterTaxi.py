import matplotlib.pyplot as plt

# Read data from the file
with open('Alpha 0.001, gamma 0.99, decay_rate 0.001.log', 'r') as file:
    lines = file.readlines()

episodes = []
rewards = []
epsilon_values = []
wrong_actions_values = []

# Parse the data
for line in lines:
    parts = line.strip().split(', ')

    #line values
    episode = int(parts[0].split(':')[3])
    reward = int(parts[1].split(':')[1])
    epsilon = float(parts[2].split(':')[1])
    wrong_action = int(parts[3].split(':')[1])

    episodes.append(episode)
    rewards.append(reward)
    epsilon_values.append(epsilon)
    wrong_actions_values.append(wrong_action)

# Create the plot
plt.subplot(2, 2, 1)
plt.plot(episodes, rewards, label='Rewards')
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(episodes, epsilon_values, label='Epsilon Deacay')
plt.ylabel('Epsilon')
plt.xlabel('Episodes')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(episodes, wrong_actions_values, label='Wrong Actions')
plt.ylabel('Wrong Actions')
plt.xlabel('Episodes')
plt.legend()

# Show the plot
plt.show()
