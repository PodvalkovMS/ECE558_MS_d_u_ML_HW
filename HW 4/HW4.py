import sys
import gym
import numpy as np
import random
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_probs(Q_s, epsilon, nA): #nA is no. of actions in the action space

    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s
  
''' 
Now we will use this get_probs func in generating the episode. 
'''
def generate_episode_from_Q(env, Q, epsilon, nA, wins, bust):
    # generates an episode from following the epsilon-greedy policy
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if next_state[0]>21 :
            bust=bust+1
        if reward>0:
            wins=wins+1
        state = next_state
        episode.append((state, action, reward))

        if done:
            break
    return episode, wins, bust



def update_Q(env, episode, Q, alpha, gamma):
    # updates the action-value function estimate using the most recent episode 
    states, actions, rewards = zip(*episode)
    # prepare for discounting

    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]] 
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q


def update_Q3(env, episode, Q, alpha, gamma):
    # updates the action-value function estimate using the most recent episode 
    states, actions, rewards = zip(*episode)
    new_reward=np.array([0 for i in range(len(rewards))])
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        if state[0]>21:
            new_reward[i]=-100
        else:
            new_reward[i]=rewards[i]
        old_Q = Q[state][actions[i]] 
        Q[state][actions[i]] = old_Q + alpha*(sum(new_reward[i:]*discounts[:-(1+i)]) - old_Q)
    return Q



def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=0.1):
    nA = env.action_space.n
    wins=0
    bust=0
    winrate=[]
    bustrate=[]
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        #epsilon = max(epsilon*eps_decay, eps_min)
        epsilon=eps_start
        # generate an episode
        episode, wins, bust = generate_episode_from_Q(env, Q, epsilon, nA, wins, bust)
        if i_episode % 25 == 0:
            winrate.append(wins/i_episode)
            bustrate.append(bust/i_episode)

        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q, winrate, bustrate


def mc_control2(env, num_episodes, alpha, gamma=1.0, eps_start=1, eps_decay=.99999, eps_min=0.001, alpha_decay=.99999, alpha_min=0.001 ):
    nA = env.action_space.n
    wins=0
    bust=0
    winrate=[]
    bustrate=[]
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        alpha = max(alpha*alpha_decay, alpha_min)
        # generate an episode
        episode, wins, bust = generate_episode_from_Q(env, Q, epsilon, nA, wins, bust)
        if i_episode % 25 == 0:
            winrate.append(wins/i_episode)
            bustrate.append(bust/i_episode)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q, winrate, bustrate

def mc_control3(env, num_episodes, alpha, gamma=1.0, eps_start=1, eps_decay=.99999, eps_min=0.001, alpha_decay=.99999, alpha_min=0.001 ):
    nA = env.action_space.n
    wins=0
    bust=0
    winrate=[]
    bustrate=[]
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        alpha = max(alpha*alpha_decay, alpha_min)
        # generate an episode
        episode, wins, bust = generate_episode_from_Q(env, Q, epsilon, nA, wins, bust)
        if i_episode % 25 == 0:
            winrate.append(wins/i_episode)
            bustrate.append(bust/i_episode)
        # update the action-value function estimate using the episode
        Q = update_Q3(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q, winrate, bustrate

def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def test_games(env, Q):
    pop=0
    n=1000
    nA=env.action_space.n
    for i_episode in range(n):
 
        state = env.reset()
        while True:
 
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], 0, nA)) \
                                        if state in Q else env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            state = next_state
            if done: 
                    if reward > 0: 
                        pop=pop+1 
                    break
    print('\n')
    winrating=str(pop/n)
    print('Winrate on the '+str(n)+' test games '+winrating)
    print('\n')
    return


env = gym.make('Blackjack-v1')
print(env.observation_space)
print(env.action_space)

policy_a, Q_a, wintate_a , bustrate_a= mc_control(env, 1000000, 0.1)

# obtain the corresponding state-value function
V_a = dict((k,np.max(v)) for k, v in Q_a.items())

#test_games(env, Q_a)

plt.plot(wintate_a)
plt.plot(bustrate_a)
# plot the state-value function
plot_blackjack_values(V_a)


policy_b, Q_b, wintate_b, bustrate_b = mc_control2(env, 10000000, 1)

# obtain the corresponding state-value function
V_b = dict((k,np.max(v)) for k, v in Q_b.items())


#test_games(env, Q_b)


plt.plot(wintate_b)
plt.plot(bustrate_b)
# plot the state-value function
plot_blackjack_values(V_b)





policy_c, Q_c, wintate_c, bustrate_c = mc_control3(env, 10000000, 1)

# obtain the corresponding state-value function
V_c = dict((k,np.max(v)) for k, v in Q_c.items())


plt.plot(bustrate_c)
plt.plot(wintate_c)

# plot the state-value function
plot_blackjack_values(V_c)

