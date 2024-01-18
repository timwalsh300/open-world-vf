# This script produced Figure 4.3 to show how much of a four-minute window
# we covered with 25,000 packets for each platform

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

protocols = ['https', 'tor']
platforms = ['youtube', 'facebook', 'vimeo', 'rumble']
platform_colors = {'youtube': 'red', 'facebook': 'blue', 'vimeo': 'gray', 'rumble': 'lightgreen'}  # Assign colors to each platform

# Determine the global x-axis range and tick marks
# Initialize variables to store the global min and max
global_min, global_max = float('inf'), float('-inf')

# First pass to find global min and max packet values
for protocol in protocols:
    for platform in platforms:
        data_list = []
        for chunk in pd.read_csv('sirinam_vf_monitored_' + protocol + '_' + platform + '.csv',
                                 header=None, chunksize=5000):
            chunk.rename(columns={25002: 'platform', 25004: 'packets'}, inplace=True)
            filtered_chunk = chunk[chunk['platform'] == platform]['packets']
            data_list.extend(filtered_chunk.values / 1000)
        global_min = min(global_min, min(data_list))
        global_max = max(global_max, max(data_list))

x_ticks = np.linspace(global_min, global_max, num=10)  # Set the number of ticks you want

# Second pass to plot
for protocol in protocols:
    for platform in platforms:
        data_list = []
        for chunk in pd.read_csv('sirinam_vf_monitored_' + protocol + '_' + platform + '.csv',
                                 header=None, chunksize=5000):
            chunk.rename(columns={25002: 'platform', 25004: 'packets'}, inplace=True)
            filtered_chunk = chunk[chunk['platform'] == platform]['packets']
            data_list.extend(filtered_chunk.values / 1000)

        data_sorted = np.sort(data_list)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        plt.plot(data_sorted, cdf, label=f'{platform.capitalize()}', color=platform_colors[platform])

    plt.axvline(x=25, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Number of Packets (Thousands)')
    plt.ylabel('CDF')
    for y in np.arange(0.1, 1.1, 0.1):
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    protocol_text = protocol.upper() if protocol == 'https' else protocol.capitalize()
    plt.title(f'CDF of Packets Per Four-Minute Capture, By Platform ({protocol_text})')
    plt.legend(loc = 'lower right')
    plt.xticks(x_ticks)
    plt.xlim([global_min, global_max])
    plt.savefig(f'packet_cdf_{protocol}.png', dpi = 300)
    plt.close()
