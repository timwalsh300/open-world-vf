import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/timothy.walsh/VF/tsne/schuster_monitored_https_oregon_vimeo_animated.csv')
features = df.drop(columns = df.columns[960:])
labels = df[df.columns[966]]
features_scaled = StandardScaler().fit_transform(features)
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(features_scaled)
df_tsne = pd.DataFrame(data=tsne_results, columns=['D1', 'D2'])
df_tsne['label'] = labels
plt.figure(figsize=(6.5, 8))
sns.scatterplot(
    x="D1", y="D2",
    hue="label",  # Color by labels
    data=df_tsne,
    palette=sns.color_palette("hls", len(df_tsne['label'].unique())),  # Adjusting palette based on the number of unique labels
    legend="full",
    alpha=0.7,
    s=20
)
plt.savefig("/home/timothy.walsh/VF/tsne/tsne_plot_monitored_https.png", dpi=300, bbox_inches='tight')
