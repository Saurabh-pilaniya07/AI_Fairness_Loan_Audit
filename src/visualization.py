import matplotlib.pyplot as plt

def plot_bias_comparison(before, after):
    labels = ['Before', 'After']
    values = [before, after]

    plt.bar(labels, values)
    plt.title("Bias Comparison")
    plt.ylabel("Demographic Parity")
    
    plt.savefig("outputs/bias_comparison.png")
    plt.close()