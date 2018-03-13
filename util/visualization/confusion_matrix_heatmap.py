import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def make_heatmap(confusion_matrix, class_names):

    """
    Adapted from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    fig = plt.figure(figsize=(15,15))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.get_cmap('Blues'))
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', )
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    plt.close()
    return data