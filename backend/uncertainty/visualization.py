
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

import matplotlib.pyplot as plt

def boxplot_analysis(correct: dict, wrong: dict, save_file: str, ylabel: str ="Probability"):

    d = []
    classes = sorted(list(correct.keys()))
    labels = []
    for k in classes:
        d += [
            correct[k],
            wrong[k]
        ]
        labels += [
            f"Correct {k}",
            f"Wrong {k}"
        ]


    # d = list(correct.values()) + list(wrong.values())
    # labels = [
    #     f"Correct {k}" for k in correct.keys()
    #     ] + [
    #     f"Wrong {k}" for k in wrong.keys()
    #     ]
    
        

    fig = plt.figure(figsize=(10,7)) 
    plt.boxplot(d, labels=labels, showfliers=False )
    
    ax = plt.gca()
    
    fig.savefig(save_file)


def boxplot_multiple_models(sorted_results_per_model: dict, save_file, title=None, xlabel=None, ylabel=None):
    if title is None:
        title = "Uncertainty per model"
    if xlabel is None:
        title = "Model"
    if ylabel is None:
        ylabel = "Uncertainty"
    
    for i, sorted_results in enumerate(sorted_results_per_model.items()):
        results = [sorted_results["correct"], sorted_results["wrong"]]

        fig = figure()
        ax = axes()

        # first boxplot pair
        bp = boxplot(results, positions = [i*2, i*2+1], widths = 0.6)
        setBoxColors(bp)

    # set axes limits and labels
    xlim(0,9)
    ax.set_xticklabels(list(sorted_results_per_model.keys()))
    ax.set_xticks([1.5 * i for i in range(len(sorted_results_per_model))])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    legend((hB, hR),('Correct', 'Wrong'))
    hB.set_visible(False)
    hR.set_visible(False)

    savefig(save_file)
    show()