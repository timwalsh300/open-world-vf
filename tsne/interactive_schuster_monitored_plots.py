import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

# for protocol in ['https', 'tor']:
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
        features = df.drop(columns = df.columns[960:])
        features_scaled = StandardScaler().fit_transform(features)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features_scaled)
        df['D1'] = tsne_results[:, 0]
        df['D2'] = tsne_results[:, 1]
        fig = px.scatter(df,
                         x = 'D1',
                         y = 'D2',
                         color = 'id',
                         hover_data=['region', 'visit', 'points'])
        pio.write_html(fig,
                       file = '/home/timothy.walsh/VF/tsne/schuster_monitored_' + protocol + '_' + platform + '.html',
                       auto_open = False)
