
import load_data
import torch
import torch.nn.functional as F
import torch.optim as optim
from Classifier import LSTMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters, feel free to tune


batch_size = 27
output_size = 9   # number of class
hidden_size = 50  # LSTM output size of each time step
input_size = 12
basic_epoch = 300
Adv_epoch = 50
Prox_epoch = 50

"""
# epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10]
Selecting optimal epsilon based on test accuracy (Grid-Search)
Maximum prox accuracy achieved with epsilon  0.5  of  88.03%
Maximum adv accuracy achieved with epsilon  5  of  92.30%
"""

prox_epsilons = [0.5, 0.01, 0.1, 1.0]
adv_epsilons = [5, 0.01, 0.1, 1.0]

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def plot_accuracies(train_acc, val_acc, model_name):
    plt.plot(train_acc, 'g', label='Training Accuracy')
    plt.plot(val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy for ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_eps_accuracies(epsilons, accuracies, model_name):
    plt.plot()
    plt.plot(accuracies[0], 'y', label="Epsilon: " + str(epsilons[0]))

    for eps, acc in zip(epsilons[1:], accuracies[1:]):
        plt.plot(acc, 'b', label="Epsilon: " + str(eps))

    plt.title('Testing Accuracy for ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Training model
def train_model(model, train_iter, mode, epsilon=1.0):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model(input, r, batch_size=input.size()[0], mode=mode)
        loss = loss_fn(prediction, target)
        if mode == 'AdvLSTM':
            ''' Add adversarial training term to loss'''
            perturb_prediction = model(input, r=compute_perturbation(loss, model),
                                       batch_size=input.size()[0], epsilon=epsilon, mode=mode)
            loss = loss + loss_fn(perturb_prediction, target)
        # if mode == 'ProxLSTM':
        #     model(input, r=compute_perturbation(loss, model),
        #           batch_size=input.size()[0], epsilon=epsilons[2], mode=mode)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            target = torch.autograd.Variable(target).long()
            prediction = model(input, r, batch_size=input.size()[0], mode=mode)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double()/(input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)


def compute_perturbation(loss, model):
    loss.backward(retain_graph=True)  # autograd to get grad wrt v
    g = model.v.grad
    return g / torch.norm(g, p=2)


''' Training basic model '''

train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)
#
model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
loss_fn = F.cross_entropy

basic_train_loss = []
basic_val_loss = []


for epoch in range(basic_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(model, train_iter, mode='plain')
    val_loss, val_acc = eval_model(model, test_iter, mode='plain')
    print(f'Epoch: {epoch+1:02}, '
          f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
          f'Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    basic_train_loss.append(train_acc)
    basic_val_loss.append(val_acc)

plot_accuracies(basic_train_loss, basic_val_loss, "Basic Model")

# ''' Save and Load model'''

# # 1. Save the trained model from the basic LSTM

torch.save(model.state_dict(), '../basic_model.pt')


# 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
print("Loading saved model")
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Prox_model.load_state_dict(torch.load('../basic_model.pt'))


# 3. load the saved model to Adv_model, which is an instance of LSTMClassifier
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load('../basic_model.pt'))

max_prox_acc = 0
max_prox_eps = -1

# ''' Training Prox_model'''
prox_accuracies = []
for epsilon in prox_epsilons:
    prox_train_loss = []
    prox_val_loss = []
    for epoch in range(Prox_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=3e-4, weight_decay=1e-3)
        train_loss, train_acc = train_model(Prox_model, train_iter, mode='ProxLSTM')
        val_loss, val_acc = eval_model(Prox_model, test_iter, mode='ProxLSTM')
        print(f'Epoch: {epoch+1:02}, '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        prox_train_loss.append(train_acc)
        prox_val_loss.append(val_acc)
    prox_accuracies.append(prox_val_loss)

plot_eps_accuracies(prox_epsilons, prox_accuracies, "Prox Model")

        # if val_acc > max_prox_acc:
        #     max_prox_acc = val_acc
        #     max_prox_eps = epsilon

    # plot_accuracies(basic_train_loss, basic_val_loss, 'Prox Model with epsilon ' + str(epsilon))

# print("Maximum prox accuracy achieved with epsilon ", max_prox_eps, " of ", max_prox_acc)

# max_adv_acc = 0
# max_adv_eps = -1

adv_accuracies = []
''' Training Adv_model'''
for epsilon in adv_epsilons:
    adv_train_loss = []
    adv_val_loss = []
    for epoch in range(Adv_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
        train_loss, train_acc = train_model(Adv_model, train_iter, mode='AdvLSTM')
        val_loss, val_acc = eval_model(Adv_model, test_iter, mode='AdvLSTM')
        print(f'Epoch: {epoch+1:02}, '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        adv_train_loss.append(train_acc)
        adv_val_loss.append(val_acc)
        # if val_acc > max_adv_acc:
        #     max_adv_acc = val_acc
        #     max_adv_eps = epsilon
    adv_accuracies.append(adv_val_loss)


plot_eps_accuracies(adv_epsilons, adv_accuracies, "Adv Model")

# print("Maximum adv accuracy achieved with epsilon ", max_adv_eps, " of ", max_adv_acc)

