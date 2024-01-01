import numpy
import pandas as pd
import sys

# the only command line argument is the number of standard deviations away from
# the mean that we're looking for

#for protocol in ['https', 'tor']:
for protocol in ['https']:
    for platform in ['youtube', 'facebook', 'vimeo', 'rumble']:
        first_columns = [str(i) for i in range(960)]
        next_columns = ['region', 'heavy_hitter', 'platform', 'genre', 'points', 'visit', 'id']
        column_names = first_columns + next_columns
        dtype_dict = {str(i): numpy.float32 for i in range(960)}
        input_csv = '/home/timothy.walsh/VF/1_csv_to_pkl/schuster_monitored_' + protocol + '_' + platform + '.csv'
        print('reading ' + input_csv + ' into a dataframe', flush = True)
        dataframe_list = []
        for chunk in pd.read_csv(input_csv,
                                 header = None,
                                 names = column_names,
                                 dtype = dtype_dict,
                                 chunksize = 5000):
            dataframe_list.append(chunk)
        df = pd.concat(dataframe_list, ignore_index = True)

        grouped_stats = df.groupby('id')['points'].agg(['mean', 'std'])

        for idx, group in df.groupby('id'):
            mean = grouped_stats.loc[idx, 'mean']
            std = grouped_stats.loc[idx, 'std']

            outliers = group[(group['points'] < mean - int(sys.argv[1]) * std) |
                             (group['points'] > mean + int(sys.argv[1]) * std)]

            for _, row in outliers.iterrows():
                print(f"Region: {row['region']}, Visit: {row['visit']}")
