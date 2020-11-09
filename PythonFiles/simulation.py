# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:36:10 2020
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
from matplotlib.lines import Line2D

path = "D:/Veena/Bentley/Fall20/MA 705/Project/Final/Dataset/"

"""
This function simulates 1000 runs of the election.
The results are 3 dataframes with index-State and column-state
"""
def simulate_election(proportion, distribution, n):
    proportions = pd.DataFrame(index = proportion.index)
    winner = pd.DataFrame(index = proportion.index)
    votes = pd.DataFrame(index = proportion.index)
    
    for k in range(n):
        sim_prop = distribution.rvs()
        # 1 indicates Biden is the winner
        # 0 indicates Trump is the winner
        sim_winner = np.where(sim_prop >= 0.505, 1, 0)
        # votes is the sum of electoral college votes that Biden wins
        sim_votes = np.where(sim_winner == 1, proportion.Votes, 0)
        
        trial_str = "Trial" + str(k+1)
        proportions[trial_str] = sim_prop
        winner[trial_str] = sim_winner
        votes[trial_str] = sim_votes
        
    return proportions, winner, votes   

proportion = pd.read_csv(path + "Output/Final_dataframe.csv")
proportion.columns = ['State','StdDev','Proportion', 'Votes']
proportion = proportion.set_index('State')

# divide proprtion by 100 to convert to percentage
distribution = stats.norm(proportion.Proportion/100, proportion.StdDev)

# 1000 simulations
num_sims = 1000

# get proportions, winner and votes from function
proportions, winner, votes = simulate_election(proportion, distribution, num_sims)
total_votes = votes.sum()
print("In this simulation, Biden wins " + str(np.sum(total_votes > 270)) + " out of " + str(num_sims) + " runs")

# count across the row (axis=1) for all 1s - 1 indicates Biden
# total_electoral_votes = num_sims
state_proportion_sum_Biden = winner.sum(axis = 1)
state_proportion_sum_Trump = num_sims - state_proportion_sum_Biden

df = pd.DataFrame(state_proportion_sum_Biden)
df['Proportion_Trump'] = state_proportion_sum_Trump
df = df.reset_index()
df.columns = ['State', 'Proportion_Biden', 'Proportion_Trump']
df = df.set_index('State')

# States in which Biden wins between 300 & 700, out of 1000 runs, are Battleground states
battleground_states = df.loc[(df['Proportion_Biden'] >= num_sims/10 *3) & (df['Proportion_Biden'] <= num_sims/10 * 7)]

print(battleground_states.index.array)


# Visualization - 1
total_votes_Biden_trial = votes.sum(axis = 0)
df1 = pd.DataFrame(total_votes_Biden_trial)
df1 = df1.reset_index()
df1.columns = ['Trial', 'Votes']
ans = df1.groupby('Votes').count()
ans = ans.sort_values(by=['Votes'])
ans = ans.reset_index()

width = 0.3
mask1 = ans['Votes'] < 270
mask2 = ans['Votes'] >= 270
p1 = plt.bar(ans['Votes'][mask1], ans['Trial'][mask1], width, align='edge', color = 'red')
p2 = plt.bar(ans['Votes'][mask2], ans['Trial'][mask2], width, align='edge', color = 'blue')

plt.ylabel('Number of trials')
plt.xlabel('Total electoral votes won by Biden')
plt.title('Number of trials vs Biden\'s electoral votes in ' + str(num_sims) + ' runs', fontsize=18)

plt.legend((p1[0], p2[0]), ('Trump', 'Biden'), fontsize=16)
plt.yticks(np.arange(0, ans['Trial'].max(), ans['Trial'].max()//10))
plt.show()

# Visualization - 2
width = 1
height = 1
nrows = 25
ncols = 40
inbetween = 0.2
color = np.where(df1['Votes'] >= 270, "blue", "red")

xx = np.arange(0, ncols, (width + inbetween))
yy = np.arange(0, nrows, (height + inbetween))

fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
ax.axis([0,ncols+1,0,nrows+1])
idx = 0

pat = []
for xi in xx:
    for yi in yy:
        sq = patches.Rectangle((xi, yi), width, height, color=color[idx], fill=True)
        ax.add_patch(sq)
        idx = idx + 1

pc = coll.PatchCollection(pat)
ax.add_collection(pc)

custom_lines = [Line2D([0], [0], color= 'red', lw=4),
                Line2D([0], [0], color= 'blue', lw=4)]
plt.legend(custom_lines, ['Trump', 'Biden'], loc=9, bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=18)

plt.title(str(num_sims) + ' election simulations', fontsize=14)
plt.axis('off')
plt.show()