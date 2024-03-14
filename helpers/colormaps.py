import seaborn as sns
from matplotlib.colors import ListedColormap


def paired(n_colors):
    return ListedColormap(sns.color_palette('Paired', n_colors=n_colors))


def seaborn_colors(n_colors):
    return ListedColormap(sns.color_palette(n_colors=n_colors))
