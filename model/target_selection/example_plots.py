import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Style
plt.style.use('fivethirtyeight')

# Create fake dataset
example_df = pd.DataFrame({'Metric A': np.array([1, 2.5, 3.5, 4, 4.25, 4, 3.5, 2.5, np.nan]),
                           'Metric B': np.array([np.nan, 1, 2.5, 3.5, 4, 4.25, 4, 3.5, 2.5]),
                           'Metric C': np.array([1.1, 2.6, 3.6, 4.1, 4.26, 4.1, 3.51, 2.51, np.nan]),
                           'Metric D': np.array([np.nan, np.nan, 1, 2.5, 3.5, 4, 4.25, 4, 3.5]),
                           'Time Period': np.arange(1, 10)})

# Plot Example Lags
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
sns.lineplot(x='Time Period', y='Metric A', data=example_df, ax=axs[0])
sns.lineplot(x='Time Period', y='Metric C', data=example_df, ax=axs[0])
axs[0].set_xticks(np.arange(1,10))
axs[0].set_ylabel('Value')
axs[0].set_xlabel('')
axs[0].set_title('Plot A')
sns.lineplot(x='Time Period', y='Metric A', data=example_df, ax=axs[1])
sns.lineplot(x='Time Period', y='Metric B', data=example_df, ax=axs[1])
axs[1].set_xticks(np.arange(1,10))
axs[1].set_title('Plot B')
sns.lineplot(x='Time Period', y='Metric A', data=example_df, ax=axs[2])
sns.lineplot(x='Time Period', y='Metric D', data=example_df, ax=axs[2])
axs[2].set_xticks(np.arange(1,10))
axs[2].set_xlabel('')
axs[2].set_title('Plot C')
plt.tight_layout()
plt.show()


# Read in data sources
player_table = pd.read_csv('../../../../data/player_ids/player_table.csv')
espn_nba_rpm = pd.read_csv('../../../../data/nba/espn/espn_nba_rpm.csv')
salary_df = pd.read_csv('../../../../data/nba/basketball_reference/player_data/salary/salary_info.csv')
bbref_player_df = pd.read_csv('../../../../data/nba/basketball_reference/player_data/combined/bbref_player_data.csv')

# Convert season from yyyy to yyyy-yyyy to join on
salary_df = salary_df[salary_df['season'].notnull()]
salary_df['season'] = salary_df.apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
espn_nba_rpm['season'] = espn_nba_rpm.apply(lambda row: str(row['season'] - 1) +  '-' +  str(row['season']), axis=1)

# Aggregatre ESPN metrics to season level to avoid problem joining traded players
espn_nba_rpm = espn_nba_rpm.groupby(['name', 'pos', 'espn_link', 'season']).mean().reset_index()

# Join dataframes
player_data = (pd.merge(bbref_player_df, player_table, how='left', left_on='BBREF_ID', right_on='bbref_id')
                        .merge(salary_df, how='left', left_on=['bbref_id', 'SEASON'], right_on=['bbref_id', 'season'])
                        .merge(espn_nba_rpm, how='left', left_on=['espn_link', 'SEASON'], right_on=['espn_link', 'season'])
                        [['BBREF_ID', 'espn_link', 'PLAYER', 'AGE', 'MP', 'SEASON', 'TEAM', 'POSITION',
                        'PER100_ORtg', 'PER100_DRtg', 'OBPM', 'DBPM', 'BPM',
                        'VORP', 'orpm', 'drpm', 'rpm', 'wins', 'salary', 'salary_prop_cap']]
                        .rename(columns={'orpm':'ORPM', 'drpm':'DRPM', 'rpm':'RPM',
                                         'wins':'WINS', 'salary':'SALARY',
                                         'salary_prop_cap':'SALARY_PROP_CAP'}))

# Create Net Rating metric
player_data['NET_RTG'] = player_data['PER100_ORtg'] - player_data['PER100_DRtg']

# Create WOR metric
player_data['WOR'] = player_data['VORP'] * 2.7

# Remove partial seasons resulting from trades.
player_data_no_trades = player_data[((player_data.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')>1) &
                        (player_data['TEAM']=='TOT')) |
                        (player_data.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')<=1)]

# Vince Carter Example
vince_carter_df = player_data_no_trades[player_data_no_trades['PLAYER']=='Vince Carter']
fig, ax = plt.subplots(figsize=(18, 5))
sns.lineplot(x=np.arange(1, 16), y='BPM', data=vince_carter_df)
ax.set_xticks(np.arange(1, 16))
ax.set_xlabel('Season')
ax2 = ax.twinx()
ax2.grid(False)
sns.lineplot(x=np.arange(1, 16), y='MP', data=vince_carter_df, ax=ax2, color='orangered')
plt.title('Vince Carter\nCareer Minutes Player vs. Box Plus-Minus')
plt.tight_layout()
plt.show()

# Kyle Korver Example
kyle_korver_df = player_data_no_trades[player_data_no_trades['PLAYER']=='Kyle Korver']
fig, ax = plt.subplots(figsize=(18, 5))
sns.lineplot(x=np.arange(1, 16), y='BPM', data=kyle_korver_df)
ax.set_xticks(np.arange(1, 16))
ax.set_xlabel('Season')
ax2 = ax.twinx()
ax2.grid(False)
sns.lineplot(x=np.arange(1, 16), y='MP', data=kyle_korver_df, ax=ax2, color='orangered')
plt.title('Kyle Korver\nCareer Minutes Player vs. Box Plus-Minus')
plt.tight_layout()
plt.show()
