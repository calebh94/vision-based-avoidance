# from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('TkAgg')

import numpy as np

import os

from avoidance_framework import DNN_C4F4F2
from train_DNN import Rescale, ToTensor, ObstacleAvoidanceDataset


def main():

    cwd = os.getcwd()
    model = DNN_C4F4F2()

    # model.load_state_dict(torch.load('Learning_models/cnn_trained_cropped_1.pt',map_location='cpu'))
    model.load_state_dict(torch.load('Models/dnn_test.pt'))

    transformations = transforms.Compose([Rescale(64), ToTensor()])

    n_testing_samples = 50
    test_sampler = SubsetRandomSampler(np.arange(n_testing_samples, dtype=np.int64))

    testing_data = ObstacleAvoidanceDataset(csv_file='data/testdata/training_data_obstacle_avoidance.csv',
                                            root_dir='data/testdata/', transform=transformations)

    test_loader = DataLoader(testing_data, batch_size=64, shuffle=False,
                                sampler=test_sampler, num_workers=4)

    model.eval()
    loss_func = torch.nn.MSELoss()
    test_loss = 0
    correct = 0

    for j, data in enumerate(test_loader):
        img_input, pose_input, expert = data['image'], data['pose'], data['astar']
        # outputs = model(pose_input[j], img_input[j])
        outputs = model(pose_input, img_input)


        img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)

        # test_loss += F.nll_loss(outputs, expert[j].float(), size_average=False).data[0]
        #
        test_loss += loss_func(outputs, expert.float())

        # pred = outputs.data.max(1,keepdim=True)[1]

        # correct += pred.eq(expert[j].float().data.view_as(pred)).sum()

        test_loss /= len(test_loader)

        print("Model Output: {}".format(outputs.data) + " \n" + "Expert Output: {}".format(expert.data))

        # output = model(data)
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        # pred = output.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset.data),
        100. * correct / len(test_loader.dataset.data)))


if __name__ == "__main__":
    main()