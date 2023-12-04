import matplotlib.pyplot as plt

def plot_figure(title, lower_bound, upper_bound, train_ratio_list, mean_accuracy_classical, mean_accuracy_quantum_32, mean_accuracy_quantum_1024):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(train_ratio_list, mean_accuracy_classical, 'r', linewidth = 2.0, label = 'Classical')
    ax1.plot(train_ratio_list, mean_accuracy_classical,'o',color = 'r')
    lns2 = ax1.plot(train_ratio_list, mean_accuracy_quantum_32, 'b', linewidth = 2.0, label = 'Quantum (32 shots)')
    ax1.plot(train_ratio_list, mean_accuracy_quantum_32,'b^')
    lns3 = ax1.plot(train_ratio_list, mean_accuracy_quantum_1024, 'indigo', linewidth = 2.0, label = 'Quantum (1024 shots)')
    ax1.plot(train_ratio_list, mean_accuracy_quantum_1024,'>', color = 'indigo')

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.set_xlim(0.29,0.91)
    ax1.set_ylim(lower_bound, upper_bound)
    ax1.set_ylabel("Classification Accuracy")
    ax1.set_xlabel("Proportion of Training Set")
    plt.title(title)
    # plt.savefig('iris.pdf', dpi=400)