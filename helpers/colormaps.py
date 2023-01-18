import seaborn as sns
from matplotlib.colors import ListedColormap

paired = lambda n_colors: ListedColormap(sns.color_palette('Paired', n_colors=n_colors))
