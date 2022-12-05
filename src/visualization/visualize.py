import matplotlib.pyplot as plt

def plot_err(errs, title, legends, x, y, save):
    if len(errs) > 0:
        for i in range(len(errs)):
            plt.plot(errs[i], label=legends[i])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend()
        plt.savefig(save)
        plt.clf()

