# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:09:21 2025

@author: maber
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
results_path = "./results/"
query_types = ['title', 'description', 'narrative']

# Load data into a dictionary of DataFrames
dfs = {}
for file in os.listdir(results_path):
    if file.endswith('.json'):
        for qtype in query_types:
            if qtype in file:
                df = pd.read_json(os.path.join(results_path, file)).T
                df['Query Type'] = qtype.capitalize()  # Add query type column
                dfs[qtype] = df

# Combine all DataFrames and reset index
combined_df = pd.concat(dfs.values()).reset_index()
combined_df.rename(columns={'index': 'Method', 'mrt': 'mrt (ms)'}, inplace=True)
combined_df['Method'] = combined_df['Method'].replace('BM25 >> MINILM % 100', 'BM25 >> SBERT % 100')


metrics = ['RR@10', 'nDCG@10', 'AP@100', 'R@10', 'R@50']
palette = {'Title': '#4C72B0', 'Description': '#55A868', 'Narrative': '#C44E52'}


#%% MAP vs query type

# Prepare data
map_df = combined_df.pivot(index='Method', columns='Query Type', values='AP@100')
map_df_T = map_df.T.reindex(['Title', 'Description', 'Narrative'])
map_df_T = map_df_T.reindex(sorted(map_df_T.columns), axis=1)

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=map_df_T, marker='o', dashes=False, markersize=8, 
             linewidth=2.5, palette='tab10')

plt.title('MAP@100 Across Query Types', fontsize=14, pad=20)
plt.xlabel('Query Type', fontsize=12)
plt.ylabel('MAP@100', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), title='Method')
plt.tight_layout()
plt.show()


#%% performance vs efficiency plots
desc_df = combined_df[combined_df['Query Type'] == 'Description']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# MAP vs Response Time
sns.scatterplot(data=desc_df, x='mrt (ms)', y='AP@100', hue='Method', 
                style='Method', s=150, ax=ax1, palette='Dark2')

ax1.set_xscale('log')
ax1.set_title('MAP@100 vs. Response Time', fontsize=13)
ax1.set_xlabel('Mean Response Time (ms, log scale)', fontsize=11)
ax1.set_ylabel('MAP@100', fontsize=11)
ax1.grid()

# RR@10 vs Response Time
sns.scatterplot(data=desc_df, x='mrt (ms)', y='RR@10', hue='Method', 
                style='Method', s=150, ax=ax2, legend=False, palette='Dark2')

ax2.set_xscale('log')
ax2.set_title('RR@10 vs. Response Time', fontsize=13)
ax2.set_xlabel('Mean Response Time (ms, log scale)', fontsize=11)
ax2.set_ylabel('Reciprocal Rank@10', fontsize=11)

ax2.grid()
plt.tight_layout()
plt.show()

#%% Recall plots
# Prepare recall data
desc_df2 = combined_df[combined_df['Query Type'] == 'Description']
recall_df2 = desc_df.melt(id_vars=['Method'], 
                        value_vars=['R@10', 'R@25', 'R@50'],
                        var_name='Recall Metric', 
                        value_name='Score')

recall_df2 = recall_df2.sort_values(by=['Method'])

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=recall_df2, x='Recall Metric', y='Score', hue='Method',
             style='Method', markers=True, dashes=False, markersize=5.5,
             linewidth=1.2, palette='tab10')

plt.title('Recall Progression Across Cutoffs (Description Queries)', fontsize=14)
plt.xlabel('Recall Metric', fontsize=12)
plt.ylabel('Recall Score', fontsize=12)
plt.ylim(0, 0.1)  # Adjust based on your data range
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()



#%% Metrics comparison bar plot
metrics = ['RR@10', 'nDCG@10', 'AP@100', 'R@50']
desc_df2.copy().set_index('Method')[metrics].plot(kind='bar', figsize=(12,6))
plt.title('Multiple Metric Comparison (Description Queries)')
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()


#%% Radar plot
from math import pi

# Prepare data with min-max normalization
metrics = ['RR@10', 'nDCG@10', 'AP@100', 'R@50']
radar_df = desc_df[['Method'] + metrics].copy()

# Min-max normalization per metric (0-1 range)
for col in metrics:
    min_val = radar_df[col].min()
    max_val = radar_df[col].max()
    radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)

# Plotting adjustments
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)
categories = metrics
N = len(categories)
angles = [n / N * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the polygon

# Style enhancements
colors = sns.color_palette("husl", len(radar_df))
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

for idx, (_, row) in enumerate(radar_df.iterrows()):
    values = row[metrics].values.flatten().tolist()
    values += values[:1]  # Close the polygon
    ax.plot(angles, values, 
            color=colors[idx],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2,
            label=row['Method'])
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# Axis adjustments
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(30)
plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
plt.ylim(0, 1)

# Legend and annotations
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
plt.title('Method Performance Profiles (Min-Max Normalized Metrics)\nDescription Queries', 
          y=1.15, fontsize=14)

plt.show()

#%% Efficiency
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Generate smoothed trend line (no strict mathematical fit required)
x = desc_df2['AP@100'].values
x = desc_df2['nDCG@10'].values
y = desc_df2['mrt (ms)'].values

# Sort data for smoothing
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

# Create smoothed curve using spline interpolation
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
y_smooth = np.linspace(y_sorted.min(), y_sorted.max(), 300)

# Create plot
plt.figure(figsize=(12, 7))
sc = sns.scatterplot(data=desc_df2, x=x, y='mrt (ms)', hue='Method',
                    size='RR@10', sizes=(50, 200), palette='viridis', 
                    edgecolor='black', alpha=0.8)

# plt.yscale('log')

exp = np.exp(np.linspace(np.log(y_sorted.min()), 
                                    np.log(y_sorted.max()), 
                                    300))

# Add to the plot after creating scatter points
plt.plot(x_smooth, exp,
        '--', color='red', linewidth=2, label='$y = e^x$')


# exp_vals = np.polyfit(x_smooth, np.log(y_smooth), 1)
# exp = np.exp(exp_vals[1])*np.exp(exp_vals[0] * x_smooth) - 90
# Add to the plot after creating scatter points
# plt.plot(x_smooth, exp,
#         '--', color='red', linewidth=2, label='$y = e^x$')

# Formatting

plt.xlabel('MAP@100', fontsize=12)
plt.xlabel('nDCG@10', fontsize=12)
plt.ylabel('MRT (ms)', fontsize=12)
plt.grid(True, alpha=0.3)
# plt.yscale('log')
plt.legend(bbox_to_anchor=(0.25, 1))

plt.tight_layout()
plt.show()

#%% 
rank_df = desc_df[metrics].rank(ascending=False)
sns.heatmap(rank_df, annot=True, cmap='viridis_r')
plt.title('Method Rankings Across Metrics (1=Best)')
plt.show()

#%% BAr plot relative to BM25
# Assuming your DataFrame has columns: ['Method', 'RR@10', 'nDCG@10', ...]
# First, set Method as index temporarily for calculations
temp_df = desc_df.set_index('Method').drop(columns = ['Query Type'])

# Get BM25 baseline values
bm25_baseline = temp_df.loc['BM25']

# Calculate percentage improvements
improvement_df = ((temp_df - bm25_baseline) / bm25_baseline * 100).reset_index()

# Clean method names (excluding BM25)
improvement_df = improvement_df[improvement_df['Method'] != 'BM25']
improvement_df['Method'] = improvement_df['Method'].str.replace('BM25 >> ', '').str.replace(' % 100', '')

# Select metrics to plot
metrics_to_plot = ['RR@10', 'nDCG@10', 'AP@100']

# Melt for better plotting
melted_df = improvement_df.melt(id_vars='Method', 
                               value_vars=metrics_to_plot,
                               var_name='Metric', 
                               value_name='Improvement')

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Method', y='Improvement', hue='Metric', 
                data=melted_df, palette=['#4C72B0', '#55A868', '#C44E52'])

plt.title('Performance Improvement Over BM25 Baseline', fontsize=14)
plt.xlabel('Re-ranking Method', fontsize=12)
plt.ylabel('Relative Improvement (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value annotations
for p in ax.patches:
    ax.annotate(f"{p.get_height():+.1f}%", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points',
                fontsize=9)

# Add reference lines
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), title='Metric')

plt.tight_layout()
plt.show()

#%% TEST
temp_df = desc_df.set_index('Method').drop(columns=['Query Type'])
bm25_baseline = temp_df.loc['BM25']

# Calculate percentage improvements (EXCLUDING MRT)
performance_metrics = ['RR@10', 'nDCG@10', 'AP@100']
improvement_df = ((temp_df[performance_metrics] - bm25_baseline[performance_metrics]) / 
                  bm25_baseline[performance_metrics] * 100).reset_index()

# Clean method names
improvement_df = improvement_df[improvement_df['Method'] != 'BM25']
improvement_df['Method'] = improvement_df['Method'].str.replace('BM25 >> ', '').str.replace(' % 100', '')

# Melt for plotting
melted_df = improvement_df.melt(id_vars='Method', 
                               value_vars=performance_metrics,
                               var_name='Metric', 
                               value_name='Improvement')

# Get response times separately
mrt_df = temp_df[['mrt (ms)']].reset_index()
mrt_df = mrt_df[mrt_df['Method'] != 'BM25']
mrt_df['Method'] = mrt_df['Method'].str.replace('BM25 >> ', '').str.replace(' % 100', '')

# Create plot
plt.figure(figsize=(14, 7))
ax = sns.barplot(x='Method', y='Improvement', hue='Metric', 
                data=melted_df, palette=['#4C72B0', '#55A868', '#C44E52'])

# Create twin axis for response times
ax2 = ax.twinx()

# Plot response times as bars
sns.barplot(x='Method', y='mrt (ms)', data=mrt_df, 
           color='#FFD700', alpha=0.7, ax=ax2)

# Configure axes
ax.set_ylabel('Relative Improvement (%)', fontsize=12)
ax2.set_ylabel('Response Time (ms, log scale)', fontsize=12)
ax2.set_yscale('log')

# Add value annotations
for p in ax.patches:
    ax.annotate(f"{p.get_height():+.1f}%", 
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points',
                fontsize=8)

for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.0f}ms", 
                 (p.get_x() + p.get_width()/2, p.get_height()),
                 ha='center', va='center', 
                 xytext=(0, 5), 
                 textcoords='offset points',
                 fontsize=8)

# Styling
plt.title('Performance Improvement vs Response Time', fontsize=14, pad=20)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(axis='y', alpha=0.3)

# Combine legends
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Rectangle((0,0),1,1, fc='#FFD700', alpha=0.7))
labels.append('Mean Response Time (ms)')
ax.legend(handles, labels, bbox_to_anchor=(1.15, 1), title='Metrics')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()