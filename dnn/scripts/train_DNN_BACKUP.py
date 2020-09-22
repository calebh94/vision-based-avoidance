# from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import os
import time
from skimage import io, transform
from avoidance_framework import DNN_C4F4F2


class ObstacleAvoidanceDataset(Dataset):
    """ Obstacle Avoidance Special Problems Dataset for Pytorch"""

    def __init__(self, csv_file, root_dir, transform=None):
        """

        :param csv_file: Path to the csv file with the image names and data
        :param root_dir: Directory with all the images
        :param transform: List of transforms to be applied on images
        """

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data.size)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = io.imread(img_name)
        pose = self.data.iloc[idx, 1:3].values
        pose = pose.astype('float').reshape(-1, 2)
        astar = self.data.iloc[idx, 14:16].values
        astar = astar.astype('float').reshape(-1, 2)
        sample = {'image': image, 'pose': pose, 'astar': astar}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pose, astar = sample['image'], sample['pose'], sample['astar']

        h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        new_h, new_w = int(self.output_size), int(self.output_size)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'pose': pose, 'astar': astar}


# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h,
#                       left: left + new_w]
#
#         landmarks = landmarks - [left, top]
#
#         return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose, astar = sample['image'], sample['pose'], sample['astar']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'pose': torch.from_numpy(pose),
                'astar': torch.from_numpy(astar)}


def show_img_batch(sample_batched):
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


def datasetadvanced():
    transformed_dataset = ObstacleAvoidanceDataset(csv_file='data/testdata/training_data_obstacle_avoidance.csv',
                                                   root_dir='data/testdata/',
                                                   transform=transforms.Compose([
                                                       Rescale(64),
                                                       ToTensor()
                                                   ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['pose'].size(), sample['astar'].size())

        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['pose'].size(),
              sample_batched['astar'].size())

        if i_batch == 3:
            plt.figure()
            show_img_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def datasettest():
    dataset_test = ObstacleAvoidanceDataset(csv_file='data/testdata/training_data_obstacle_avoidance.csv',
                                            root_dir='data/testdata/')
    fig = plt.figure()

    for i in range(len(dataset_test)):
        sample = dataset_test[i]

        print(i, sample['image'].shape, sample['pose'].shape, sample['astar'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_image_annotated(**sample)

        if (i == 3):
            plt.show()
            break

    # define transforms
    scale = Rescale(64)

    fig2 = plt.figure()
    sample1 = dataset_test[2]
    for i, tsfrm in enumerate([scale]):
        tf_sample = tsfrm(sample1)

        ax = plt.subplot(1, 1, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_image_annotated(**tf_sample)

    plt.show()


def show_image_annotated(image, pose, astar):
    plt.imshow(image)
    annotation_string = "Poses: {}".format(pose[:]) + " \n" + "Astar: {}".format(astar[:])
    plt.text(0, 600, annotation_string, verticalalignment='bottom', horizontalalignment='left',
             color='red')
    plt.pause(0.001)


def dataprep():
    # Reading data from csv with annotations being the
    frame_and_data = pd.read_csv('data/testdata/training_data_obstacle_avoidance.csv')

    n = 65
    img_name = frame_and_data.iloc[n, 0]
    pose = frame_and_data.iloc[n, 1:3].values
    pose = pose.astype('float').reshape(-1, 2)
    pose_x = frame_and_data.iloc[n, 1]
    pose_y = frame_and_data.iloc[n, 2]
    astar = frame_and_data.iloc[n, 14:16].values
    astar = astar.astype('float').reshape(-1, 2)
    astar_x = frame_and_data.iloc[n, 14]
    astar_y = frame_and_data.iloc[n, 15]

    print("Image name: {}".format(img_name))
    print("Pose X: {}".format(pose_x))
    print("Pose Y: {}".format(pose_y))
    print("Pose Shape: {}".format(pose.shape))
    print("Astar X: {}".format(astar_x))
    print("Astar y: {}".format(astar_y))
    print("Astar Shape: {}".format(astar.shape))

    print("Poses: {}".format(pose[:]))
    print("Astar: {}".format(astar[:]))

    plt.figure()
    show_image_annotated(io.imread(os.path.join('data/data4-4/', img_name)), pose, astar)
    plt.show()


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss Function
    # loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


def TrainNet(net, batch_size, n_epochs, learning_rate, train_loader):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    # train_loader = get_train_loader(batch_size, training_set, train_sampler)
    n_batches = enumerate(train_loader).__sizeof__()
    # n_batches = 2

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches / 10
        start_time = time.time()
        total_train_loss = 0

        # for i, data in enumerate(train_loader, 0):
        for i, data in enumerate(train_loader):

            # Get inputs
            # inputs, labels = data
            img_input, pose_input, expert = data['image'], data['pose'], data['astar']

            # Wrap them in a Variable object
            # inputs, labels = Variable(inputs), Variable(labels)
            img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)

            if next(net.parameters()).is_cuda == True:
                img_input = img_input.to(torch.device("cuda"))
                pose_input = pose_input.to(torch.device("cuda"))
                expert = expert.to(torch.device("cuda"))

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            # outputs = net(pose_input[i], img_input[i])
            outputs = net(pose_input, img_input)

            loss_size = loss(outputs, expert[i].float())
            loss_size = loss(outputs, expert.float())

            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every,
                    time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        # total_val_loss = 0
        # for inputs, labels in validation_loader:
        #     # Wrap tensors in Variables
        #     inputs, labels = Variable(inputs), Variable(labels)
        #
        #     # Forward pass
        #     val_outputs = net(inputs)
        #     val_loss_size = loss(val_outputs, labels)
        #     total_val_loss += val_loss_size.data[0]
        #
        # print("Validation loss = {:.2f}".format(total_val_loss / len(validation_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


# def TestNet(model, batch_size, n_epochs, learning_rate, test_loader):
#
#
#
#     for j, data in enumerate(test_loader):
#         img_input, pose_input, expert = data['image'], data['pose'], data['astar']
#         # outputs = model(pose_input[j], img_input[j])
#         outputs = model(pose_input, img_input)
#
#
#         img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)
#
#         # test_loss += F.nll_loss(outputs, expert[j].float(), size_average=False).data[0]
#         #
#         test_loss += loss_func(outputs, expert.float())
#
#         # pred = outputs.data.max(1,keepdim=True)[1]
#
#         # correct += pred.eq(expert[j].float().data.view_as(pred)).sum()
#
#         test_loss /= len(test_loader)
#
#         print("Model Output: {}".format(outputs.data) + " \n" + "Expert Output: {}".format(expert.data))
#
#         # output = model(data)
#         # test_loss += F.nll_loss(output, target, size_average=False).data[0]
#         # get the index of the max log-probability
#         # pred = output.data.max(1, keepdim=True)[1]
#         # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#         # test_loss /= len(test_loader.dataset)
#     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#         test_loss, correct, len(test_loader.dataset.data),
#         100. * correct / len(test_loader.dataset.data)))


def main():
    # Initialize model
    model = DNN_C4F4F2().cuda()

    # define parameters
    batch_size_train = 64

    # define transforms
    transformations = transforms.Compose([Rescale(64), ToTensor()])

    # Training
    n_training_samples = 3000
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    # Testing dataset

    training_data = ObstacleAvoidanceDataset(csv_file='data/data4-4/training_data_obstacle_avoidance.csv',
                                             root_dir='data/data4-4/', transform=transformations)
    training_loader = DataLoader(training_data, batch_size=batch_size_train, shuffle=False,
                                 sampler=train_sampler, num_workers=4)

    # validation_loader =

    TrainNet(model, batch_size=batch_size_train, n_epochs=50, learning_rate=0.0005, train_loader=training_loader)
    torch.save(model.state_dict(), 'Models/dnn_4-4_v1.pt')


if __name__ == "__main__":
    main()
    # dataprep()
    # datasettest()
    # datasetadvanced()


