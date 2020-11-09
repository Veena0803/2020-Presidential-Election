# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:20:24 2020
"""

import pandas as pd
import glob
import numpy as np
import geopandas as gpd
from pollster_name_changes import pollster_changes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Change this to the path where you place the entire zip
path = "D:/Veena/Bentley/Fall20/MA 705/Project/Final/Dataset/"

# Pollster variables
pollster_csv = pd.read_csv(path + 'Input/pollster-ratings.csv', header = 0)
pollster_df = pollster_csv[['Pollster', '538 Grade', 'Bias']]

# Electoral votes
elector = pd.DataFrame()
elector = pd.read_csv(path + "Input/Electoral.txt", sep='\t',engine='python')

# Master dataframe
df = pd.DataFrame()

for filename in glob.iglob(path + 'Input/StatePolls/*.txt', recursive = True): 
   
    # Getting name of State
    name=filename.split('/')[-1]    
    name=name.split('\\')[-1]
    state = name[:-4]
    df1 = pd.read_csv(filename, sep='\t',engine='python')
    # Adding column for state
    df1['State']=state
    # Removing %
    df1.Other = df1.Other.str.replace('[%]', '')
    df1.Trump = df1.Trump.str.replace('[%]', '')
    df1.Biden = df1.Biden.str.replace('[%]', '')
    # Converting to type float
    df1["Other"] = df1["Other"].astype(float)
    df1["Biden"] = df1["Biden"].astype(float)
    df1["Trump"] = df1["Trump"].astype(float)
    # Distributing the Other equally between Biden and Trump
    df1["Biden"] = df1["Biden"]+(df1["Other"]/2)
    df1["Trump"] = df1["Trump"]+(df1["Other"]/2)
    # Getting sample size and converting to float
    df1.Sample = df1.Sample.str.split()
    df1.Sample = df1.Sample.str[0]
    df1.Sample = df1.Sample.str.replace('[,]', '')
    df1["Sample"] = df1["Sample"].astype(float)
    # Estimated p hat Biden wins
    df1['p'] = (df1["Biden"]/100)
    # Estimated Std Deviation
    df1['s'] = np.sqrt(((df1['p']*(1-df1['p']))/(df1['Sample'])))
    df = df.append(df1)
    
    # Filter out 2020 rows
    year = df.Date.str.split('/')
    year = year.str[2].astype(int)
    is_2020 = year == 2020
    df = df[is_2020]
    
    grade = []
    bias = []
    
    # Merge with Pollster data
    for index, row in df.iterrows():
     # access data using column names
     # print(index, row['delay'], row['distance'], row['origin'])
        # lookup poll_source in key_value dictionary
        poll_name = pollster_changes.get(row['Source'])
        
        if poll_name:            
            # locate row in pollster data
            poll_info = pollster_df.loc[pollster_df.Pollster == poll_name, :]
            # print(poll_info['538 Grade'])
            # add new columns Grade and Bias 
            grade.append(poll_info['538 Grade'].iloc[0])
            bias.append(poll_info['Bias'].iloc[0])
            # print(row)
        else:
            grade.append("0")
            bias.append("0")
    
    df['Grade'] = grade
    df['Bias'] = bias
    
    
df = df[['State','Source','Date','Sample','Biden','Trump','p','s', 'Grade', 'Bias']]
df.columns = ['State','Poll Source','Poll Date','Sample Size','Reported Proportion for Biden','Reported Proportion for Trump', 'Estimated p hat Biden wins','Estimated std dev', 'Grade', 'Bias']

df.to_csv (path + 'Output/Merged_dataframe.csv', index = False, header=True)
df.head()

# Extracting columns without blank values
data = pd.read_csv(path + "Output/Merged_Dataframe.csv")
new_df = data.dropna(axis = 'index', how = 'any')

#Spliting the Bias column using the '+' separator
bias_eliminate = new_df.Bias.str.split('+')

# Segregating data as democrat or republican based on 'D' or 'R' which is included in the Bias column
bias_d = []
for bias in bias_eliminate.array:
    if(bias[0].strip() == 'D'):
        democrat_bias = bias[1]
    else:
        democrat_bias = 0
    bias_d.append(democrat_bias)
new_df['Democrat Bias'] = bias_d

bias_r = []
for bias in bias_eliminate.array:
    if(bias[0].strip() == 'R'):
        republican_bias = bias[1]
    else:
        republican_bias = 0
    bias_r.append(republican_bias)
new_df['Republican Bias'] = bias_r
        
# Converting Democrat Bias to float
new_df['Democrat Bias'] = new_df['Democrat Bias'].astype(float)

# Calculating the mean proportion for Biden grouped by State
avg = new_df.groupby('State')['Reported Proportion for Biden'].mean()
avg_dem_bias = new_df.groupby('State')['Democrat Bias'].mean()

avg_biden = avg - avg_dem_bias

avg_sd = new_df.groupby('State')['Estimated std dev'].mean()

df2 = pd.DataFrame(data = avg_sd)
df2 = df2.reset_index()
df2.columns=['State','Avg Std Dev']

df2.State = df2.State.str.lower()
df2.State = df2.State.str.title()
df3 = pd.DataFrame(data = avg_biden)
df3 = df3.reset_index()
df3.columns=['State','Avg_Proportion']

df3.State = df3.State.str.lower()
df3.State = df3.State.str.title()
total = pd.merge(df2, df3, on= 'State')

total.columns=['State','Avg Std Dev','Avg_Proportion']
total=total.set_index('State')

#  Sorting by state alphabetically 
total = total.sort_values(by=['State'])
total = total.reset_index()

# Objective here is to compute final electoral score for Biden and Trump
# Get the electoral votes and Avg polls for Biden for each state in one dataframe and if avg >= 50, electoral votes go to Democrats

# Get number of rows from elector dataframe
rows=elector.shape[0]
# Convert from 4 columns to 2 columns
elector=elector.values.reshape(rows*2,2)
# Converting to dataframe
elector = pd.DataFrame(data = elector)
# Drop rows with NaN
elector = elector.dropna(axis=0, how='any')
# Naming colmuns
elector.columns = ['State','Electoral Votes']

# Removing spaces between names of states for ex: New Hampshire to Newhampshire
elector.State = elector.State.str.replace(' ', '')
elector.State = elector.State.str.lower()
elector.State = elector.State.str.title()
elector= elector.sort_values(by=['State'])

total.State = total.State.str.replace(' ', '')
total.State = total.State.str.lower()
total.State = total.State.str.title()
total = total.sort_values(by=['State'])

# Merging elector df with total
elec_avg_df = pd.merge(total, elector, on= 'State')
elec_avg_df.to_csv(path + 'Output/Final_dataframe.csv', index = False, header=True)

# For each state, if avg >= 50, electoral votes go to Democrats, else go to Republicans
Blue = Red = 0
for i in elec_avg_df.index:
    elector['Color']=np.where(elec_avg_df['Avg_Proportion']>=50,"Blue","Red")
    
for i in elec_avg_df.index:
    if (elec_avg_df['Avg_Proportion'].iloc[i] >=50):
        Blue = Blue + elec_avg_df['Electoral Votes'].iloc[i]
    else:
        Red = Red + elec_avg_df['Electoral Votes'].iloc[i]
print("Final Electoral Votes for Biden is", Blue)
print("Final Electoral Votes for Trump is", Red)

# Bar graph showing final electoral score
color = ("Blue", "Red")
data = [Blue, Red]    
plt.bar(range(1,3,1),data,width =0.2, color = color)
plt.ylabel("Electoral Votes", fontsize = 16)
x_labels = ["Biden","Trump"]
plt.xticks(range(1,3,1),labels=x_labels)
plt.title("Final Electoral Score for Biden and Trump ", fontsize = 20)


states = gpd.read_file(path + 'Map/states.shp')
states.sort_values('STATE_NAME')
states = states.rename(columns = {'STATE_NAME': 'State'})
states.State = states.State.str.replace(' ', '')
states.State = states.State.str.lower()
states.State = states.State.str.title()

states_avg = pd.merge(states, elec_avg_df, on='State', how='inner')
#convert to Mercator projection
#states = states.to_crs("EPSG:3395")
#for state in states_avg.columns.to_list():
states_avg_without_alaska_hi = states_avg[states_avg.State.isin(['Alaska', 'Hawaii']) == False]
continental = states[states.STATE_ABBR.isin(['AK', 'HI']) == False]

custom_lines = [Line2D([0], [0], color='orangered', lw=4),
            Line2D([0], [0], color='coral', lw=4),
            Line2D([0], [0], color='grey', lw=4),
            Line2D([0], [0], color='lightskyblue', lw=4),
            Line2D([0], [0], color='dodgerblue', lw=4)]

#lines = ax.plot(data)

state_colors = ['orangered', 'coral', 'grey', 'lightskyblue', 'dodgerblue' ]
line_widths = [1, .5, .5, 1, .5,.5]

fig, ax = plt.subplots(figsize=(14, 14))

states_avg_without_alaska_hi.plot(
                column = 'Avg_Proportion',
                ax=ax, 
                edgecolor = "black",
                cmap=ListedColormap(state_colors),
                legend=True)
ax.legend(custom_lines, ['Safe Republican', 'Lean Republican', 'Battleground', 'Lean Democrat', 'Safe Democrat'], fontsize = 20, loc = 'lower right')

for state in continental.index:
    ax.annotate(s=continental.STATE_ABBR[state], xy = (continental.geometry[state].centroid.x,
                                                         continental.geometry[state].centroid.y),
                ha = 'center', fontsize=10)
plt.title('Statewise predicted winner', fontsize = 20)
plt.axis('off')

ax_AK = inset_axes(ax, width="25%", height="25%", loc="lower left", borderpad=0)
alaska = states_avg[states_avg['State'] == 'Alaska']
alaska.plot(column = 'Avg_Proportion', edgecolor = "grey", ax=ax_AK, cmap=ListedColormap(state_colors))
ax_AK.annotate(s="AK", xy = (alaska.geometry.centroid.x, 
                                alaska.geometry.centroid.y), ha = 'center', fontsize=10)
ax_AK.set_xticks([])
ax_AK.set_yticks([])
ax_AK.axis('off')

ax_HI = inset_axes(ax, bbox_to_anchor=(.2, 0, 1, 1),
                   bbox_transform=ax.transAxes, width="15%", height="15%", loc="lower left", borderpad=0)
hawaii = states_avg[states_avg['State'] == 'Hawaii']
hawaii.plot(column = 'Avg_Proportion', edgecolor = "grey", ax=ax_HI, color="dodgerblue")
ax_HI.annotate(s="HI", xy = (hawaii.geometry.centroid.x, 
                                hawaii.geometry.centroid.y), ha = 'center', fontsize=10)
ax_HI.set_xticks([])
ax_HI.set_yticks([])
ax_HI.axis('off')
