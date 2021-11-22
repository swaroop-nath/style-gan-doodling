import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def structure_data(single_data):
    final_data_point = []
    final_data_point.append(eval(single_data[0].split(': ')[1].split(' - ')[0]))
    final_data_point.append(eval(single_data[0].split(': ')[-1]))
    final_data_point.append(eval(single_data[1].split(': ')[-1]))
    final_data_point.append(eval(single_data[3].split(': ')[-1]))

    return final_data_point

def structure_alphas(single_data):
    return eval(single_data[2].split(': ')[-1])

def process_data(file_path):
    with open(file_path, 'r') as file:
        contents = file.read()

    iter_data = contents.split('\n')

    iter_data_split = [single_iter.split('\t|') for single_iter in iter_data]

    structured_losses_with_steps = [structure_data(single_iter) for single_iter in iter_data_split]
    alphas = [structure_alphas(single_iter) for single_iter in iter_data_split]
    columns = ['Iteration', 'Gen Loss', 'Critic Loss', 'Step']
    loss_data = pd.DataFrame(structured_losses_with_steps, columns=columns)

    return loss_data, alphas

def generate_plots(loss_data, alphas, file_name):
    plt.figure()
    plt.ylim([0, 50])
    loss_plot = sns.lineplot(data=loss_data, x='Iteration', y='Gen Loss', hue='Step').get_figure()
    loss_plot.savefig(file_name + '_gen_losses.png')
    plt.figure()
    plt.ylim([0, 2])
    loss_plot = sns.lineplot(data=loss_data, x='Iteration', y='Critic Loss', hue='Step').get_figure()
    loss_plot.savefig(file_name + '_critic_losses.png')
    plt.figure()
    plt.ylim([0, 1.1])
    alpha_plot = sns.lineplot(x=list(range(len(alphas))), y=alphas).get_figure()
    alpha_plot.savefig(file_name + '_alphas.png')


if __name__ == '__main__':
    for idx, file_name in enumerate(['loss_1.log', 'loss_2.log']):
        loss_data, alphas = process_data(file_name)
        generate_plots(loss_data, alphas, 'plot_{}'.format(idx + 1))