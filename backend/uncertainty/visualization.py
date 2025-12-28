
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes
import numpy as np

def set_boxplot_line_color(bp, color):
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color(color)

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

def boxplot_analysis(correct: dict, wrong: dict, save_file: str, metric: str, title: str = ""):

    classes = sorted(list(correct.keys()))

    correct_data = []
    wrong_data = []
    for k in classes:
        correct_data.append(correct[k])
        wrong_data.append(wrong[k])

    num_groups = len(classes)
    positions = np.arange(num_groups)

    offset = 0.15
    width = 0.2
    
    fig = plt.figure(figsize=(10,7))
    ax = plt.gca()

    
    bp_correct = plt.boxplot(
        correct_data,
        positions=positions - offset,
        widths=width,
        patch_artist=False,
        showfliers=False
    )

    bp_wrong = plt.boxplot(
        wrong_data,
        positions=positions + offset,
        widths=width,
        patch_artist=False,
        showfliers=False
    )

    # Color the boxes
    set_boxplot_line_color(bp_correct, "blue")
    set_boxplot_line_color(bp_wrong, "red")

    # Axis formatting
    plt.xticks(positions, classes)
    plt.ylabel(metric.capitalize())
    plt.legend(
        [bp_correct['boxes'][0], bp_wrong['boxes'][0]],
        ['Correct', 'Wrong']
    )
    
    ax.set_xlabel("Response prediction")
    ax.set_title(title)
    
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