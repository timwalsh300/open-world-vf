import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy

first_columns = [str(i) for i in range(7680)]
next_columns = ['region',
                'heavy_hitter',
                'platform',
                'genre',
                'points',
                'visit',
                'id']
column_names = first_columns + next_columns
dtype_dict = {str(i): numpy.float32 for i in range(7680)}

def df_to_tsne(df, version):
    df = df.reset_index(drop=True)
    features = df.drop(columns = df.columns[7680:])
    labels = df['id']
    color_palette = {
        60: '#ff6666',
        180: '#b3e6ff',
        240: '#666666',
        241: '#00cc00'
    }
    features_scaled = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features_scaled)
    df_tsne = pandas.DataFrame(data=tsne_results, columns=['D1', 'D2'])
    df_tsne['label'] = labels
    plt.figure(figsize=(6.5, 8))
    sns.scatterplot(
        x='D1', y='D2',
        hue='label',  # Color by labels
        data=df_tsne,
        palette=color_palette,
        legend='full',
        alpha=0.7,
        s=20
    )
    plt.savefig('tsne_plot_' + version + '.png', dpi=300, bbox_inches='tight')

def add_noise(row):
    # Sample a random standard deviation for the current row
    stddev = numpy.random.uniform(0.01, 0.1)
    # Generate noise with zero mean and the sampled stddev
    noise = numpy.random.normal(0, stddev, size=row.shape)
    return row + noise

# read in the monitored CSV in chunks and concatenate them together
dataframe_list = []
for chunk in pandas.read_csv('../3_open_world_baseline/dschuster16_monitored_https.csv',
                             header = None,
                             names = column_names,
                             dtype = dtype_dict,
                             chunksize = 5000):
    dataframe_list.append(chunk)
monitored = pandas.concat(dataframe_list, ignore_index = True)

# get groups of classes and add the first 50 instances of class A
# and class B to a pair of dataframes called sources and targets
monitored_grouped = monitored.groupby('id')
sources = pandas.DataFrame(columns = column_names)
targets = pandas.DataFrame(columns = column_names)
for name, group in monitored_grouped:
    if name == 60:
        temp = group.head(80)
        sources = pandas.concat([sources, temp])
    if name == 180:
        temp = group.head(80)
        targets = pandas.concat([targets, temp])
# concatenate sources and targets and produce a plot showing
# their relationship to each other
monitored_only = pandas.concat([sources, targets])
monitored_only = monitored_only.drop(columns = monitored_only.columns[7680:7686])
# df_to_tsne(monitored_only, 'monitored_only')

# initialize a dataframe to hold uniform padding instances
uniform_padding = pandas.DataFrame(columns = first_columns)
# do this twice so that we end up with a number of uniform padding
# instances equal to the size of monitored_only
for i in range(2):
    # shuffle and drop labels
    sources = sources.sample(frac = 1).reset_index(drop = True).drop(columns = sources.columns[7680:])
    targets = targets.sample(frac = 1).reset_index(drop = True).drop(columns = targets.columns[7680:])
    # select a different lambda for each pair
    lambdas = numpy.random.uniform(0.2, 0.8, size=sources.shape[0])
    # compute the weighted averages for each pair based on their lambda
    wavg_x = lambdas[:, None] * sources + (1 - lambdas[:, None]) * targets
    uniform_padding = pandas.concat([uniform_padding, wavg_x])
# add NOTA labels to the uniform padding instances
uniform_padding['id'] = 241

# initialize a dataframe to hold mean padding instances
mean_padding = pandas.DataFrame(columns = first_columns)
# do this twice so that we end up with a number of mean padding
# instances equal to the size of monitored_only
for i in range(2):
    # shuffle and drop labels
    sources = sources.sample(frac = 1).reset_index(drop = True).drop(columns = sources.columns[7680:])
    targets = targets.sample(frac = 1).reset_index(drop = True).drop(columns = targets.columns[7680:])
    # compute the mean for each pair
    mean_x = 0.5 * sources + 0.5 * targets
    mean_x_noisy = mean_x.apply(add_noise, axis=1)
    mean_padding = pandas.concat([mean_padding, mean_x_noisy])
# add NOTA labels to the mean padding instances
mean_padding['id'] = 241
monitored_uniform_mean = pandas.concat([monitored_only, uniform_padding, mean_padding])
df_to_tsne(monitored_uniform_mean, 'monitored_nota')

# read in the unmonitored CSV in chunks and concatenate them together
dataframe_list = []
for chunk in pandas.read_csv('../3_open_world_baseline/dschuster16_unmonitored_https.csv',
                             header = None,
                             names = column_names,
                             dtype = dtype_dict,
                             chunksize = 100):
    dataframe_list.append(chunk)
    break
unmonitored = pandas.concat(dataframe_list, ignore_index = True)
unmonitored = unmonitored.drop(columns = unmonitored.columns[7680:7686])
monitored_uniform_mean_unmonitored = pandas.concat([unmonitored, monitored_uniform_mean])
df_to_tsne(monitored_uniform_mean_unmonitored, 'monitored_nota_unmonitored')

monitored_unmonitored = pandas.concat([unmonitored, monitored_only])
df_to_tsne(monitored_unmonitored, 'monitored_unmonitored')
