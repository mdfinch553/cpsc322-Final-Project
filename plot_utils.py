import matplotlib.pyplot as plt
import math

def plot_bar_chart(x, y):
    if len(x) > 10:
        plt.figure(figsize=(18,15))
    else:
        plt.figure()
    plt.bar(x, y)
    plt.xticks(x, rotation=45, horizontalalignment="right", size="small")
    plt.show()

def plot_pie_chart(x, y):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()

def plot_histogram(data):
    # data is a 1D list of values
    plt.figure() 
    plt.hist(data, bins=10) # bins=10    

    plt.show() 

def plot_two_histograms(data1, data2, label1, label2):
    plt.figure() 
    plt.hist(data1, bins=10, label=label1) # bins=5    
    plt.hist(data2, bins=10, label=label2) # bins=5
    plt.legend()
    plt.show()     

def plot_scatter(x, y):
    plt.figure()
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    cov = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / (len(x) -1)
    corr = (sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])) / (math.sqrt(sum([(x[i] - mean_x) **2 for i in range(len(x))])) * math.sqrt(sum([(y[i] - mean_y) ** 2 for i in range(len(y))])))
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
        / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    b = mean_y - m * mean_x
    plt.scatter(x, y, label=f'Correlation ={corr}\n Covariance ={cov}')
    plt.legend()
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="m", lw=5)
    plt.show()

def plot_box_plots(distributions, labels):
    plt.figure(figsize = (15, 3))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation=45, horizontalalignment="right", size="small")
    plt.show()