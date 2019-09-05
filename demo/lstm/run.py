# -*- coding: utf-8 -*-
import os
import json
import time
import math
import matplotlib.pyplot as plt
from demo.lstm.data_loader import DataLoader
from demo.lstm.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # 填充预测列表以将其在图表中移动到正确的开始
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)

    # x, y = data.get_train_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )
    # in-memory training
    # model.train(
    #     x,
    #     y,
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     save_dir=configs['model']['save_dir']
    # )

    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # 多序列预测
    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
                                                   configs['data']['sequence_length'])

    print(predictions)
    # 绘图
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

    # 全序列预测
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # 绘图
    # plot_results(predictions, y_test)

    # 逐点预测
    # predictions = model.predict_point_by_point(x_test)
    # # 绘图
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
