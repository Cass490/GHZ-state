from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False})


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_confusion(matrix, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Separable', 'Entangled'],
           yticklabels=['Separable', 'Entangled'], xlabel='Predicted', ylabel='Actual', title=title)
    save_figure(fig, path)


def plot_study(df, x, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(['accuracy', 'f1'], axes):
        for model, rows in df.groupby('model'):
            grouped = rows.groupby(x)[metric].agg(['mean', 'std']).fillna(0)
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=3, label=model)
        ax.set(xlabel='Pauli observable count' if x == 'num_observables' else 'State noise p',
               ylabel=metric.upper() if metric == 'f1' else 'Accuracy', ylim=(0, 1.05))
        ax.legend()
    fig.suptitle(title)
    save_figure(fig, path)


def plot_dataset(df, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    table = df.groupby(['family', 'label']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    table.rename(columns={0: 'Separable', 1: 'Entangled'}).plot.bar(ax=ax, rot=0)
    ax.set(xlabel='Original state family', ylabel='Reduced-state samples', title='Labels determined by PPT, across noise variants')
    save_figure(fig, path)
