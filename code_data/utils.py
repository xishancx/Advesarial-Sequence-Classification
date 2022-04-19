import matplotlib.pyplot as plt
from matplotlib import colors


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def plot_accuracies(train_acc, val_acc, model_name):
    plt.plot(train_acc, 'g', label='Training Accuracy')
    plt.plot(val_acc, 'b', label='Testing Accuracy')
    plt.title('Training and Testing Accuracy for ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_eps_accuracies(epsilons, accuracies, model_name):
    plt.plot()
    # plt.plot(accuracies[0], c=epsilons, label="Epsilon: " + str(epsilons[0]))

    for eps, acc, c in zip(epsilons, accuracies, colors.BASE_COLORS):
        plt.plot(acc, c=c, label="Epsilon: " + str(eps))

    plt.title('Testing Accuracy for ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
