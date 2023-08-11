import matplotlib.pyplot as plt

from itertools import cycle
from wordcloud import WordCloud

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


colors = 'bgrcmykbgrcmykbgrcmykbgrcmyk'


def show_wordcloud(source, max_words=50):
    try:
        wordcloud = WordCloud(max_words=1000)
        if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
            wordcloud.generate_from_text(source)
        else:
            wordcloud.generate_from_frequencies(source)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        raise ValueError("Invalid data type for source parameter: str or [(str,float)]")
           
            
            
def show_clusters_high_dim(model, X, method='pca', title=''):
    method = method.lower().strip()
    if method == 'pca':
        reduced_data = PCA(n_components=2).fit_transform(X)
    elif method == 'tsne':
        reduced_data = TSNE(n_components=2).fit_transform(X)
    else:
        raise ValueError("Invalid data type for method parameter: 'pca' or 'tsne'")
    # Get the labels from model
    labels = model.labels_
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    for idx, instance in enumerate(reduced_data):
        label = labels[idx]
        color = colors[label]
        ax.plot(instance[0], instance[1], marker='o', color=color, linestyle='', ms=15, mec='none')
        ax.set_aspect('auto')
    
    plt.show() #show the plot
            
            
            
#
# Everything below gets only executed when the file is explicitly being run
# and not when imported. This is useful for testing the functions.
#
if __name__ == "__main__":
    
    source = "hi hi hi hi hi hi ho ho ho ho ha"
    
    show_wordcloud(source)