# from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.optim as optim
from torch.autograd import Variable
from visdom import Visdom

from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import os
import time
from skimage import io, transform
from skimage.color import rgb2gray
from dnn.scripts.dnn_architecture import DNN_C4F4F2_Astar_small, DNN_C4F4F2_Astar


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        # img_name = os.path.join(self.root_dir, self.data.iloc[idx,0])
        img_name = self.data.iloc[idx,0]
        image = io.imread(img_name)
        pose = self.data.iloc[idx, 1:3].values
        pose = pose.astype('float').reshape(-1, 2)
        astar = self.data.iloc[idx, 14:16].values
        astar = astar.astype('float').reshape(-1, 2)
        sample = {'image': image, 'pose': pose, 'astar': astar}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ObstacleAvoidanceDataset_new(Dataset):
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
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        # img_name = os.path.join(self.root_dir, self.data.iloc[idx,0])
        img_name = str(self.data.iloc[idx,1])
        image = io.imread(os.path.join(self.root_dir,img_name))
        pose = self.data.iloc[idx, 2:4].values
        pose = pose.astype('float').reshape(-1, 2)
        astar = self.data.iloc[idx, 15:17].values
        astar = astar.astype('float').reshape(-1, 2)
        sample = {'image': image, 'pose': pose, 'astar': astar}

        if self.transform:
            sample = self.transform(sample)

        return sample


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y.cpu().numpy(),y.cpu().numpy()]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y.cpu().numpy()]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


class Grayscale(object):

    def __call__(self, sample):
        img, pose, astar = sample['image'], sample['pose'], sample['astar']
        img = rgb2gray(img)
        img = np.expand_dims(img, 2)

        return {'image': img, 'pose': pose, 'astar': astar}


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


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, pose, astar = sample['image'], sample['pose'], sample['astar']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # top = np.random.randint(0, h - new_h)
        top = h - new_h
        # left = np.random.randint(0, w - new_w)
        left = w - new_w

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'pose': pose, 'astar': astar}


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


class ToTensor_Norm(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose, astar = sample['image'], sample['pose'], sample['astar']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': F.normalize(torch.from_numpy(image),3),
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
    transformed_dataset = ObstacleAvoidanceDataset(csv_file='/home/charris/Desktop/data_collector/2019-04-06_13-50-14/training_data_obstacle_avoidance_cleaned.csv',
                                            root_dir='/home/charris/Desktop/data_collector/2019-04-06_13-50-14/',
                                                   transform=transforms.Compose([
                                                       Crop((360,640)),
                                                       Rescale(128),
                                                       # Grayscale(),
                                                       Grayscale(),
                                                       ToTensor_Norm()
                                                   ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['pose'].size(), sample['astar'].size())

        if i==3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True,num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['pose'].size(),
              sample_batched['astar'].size())

        if i_batch==3:
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

        print(i, sample['image'].shape, sample['pose'].shape,sample['astar'].shape)

        ax = plt.subplot(1,4,i+1)
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
    for i,tsfrm in enumerate([scale]):
        tf_sample = tsfrm(sample1)

        ax = plt.subplot(1,1,i+1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_image_annotated(**tf_sample)

    plt.show()


def show_image_annotated(image, pose, astar):
    plt.imshow(image)
    annotation_string = "Poses: {}".format(pose[:]) + " \n" + "Astar: {}".format(astar[:])
    plt.text(0,600, annotation_string, verticalalignment='bottom', horizontalalignment='left',
             color='red')
    plt.pause(0.001)


def dataprep():
    # Reading data from csv with annotations being the
    frame_and_data = pd.read_csv('data/testdata/training_data_obstacle_avoidance.csv')

    n = 65
    img_name = frame_and_data.iloc[n,0]
    pose = frame_and_data.iloc[n,1:3].values
    pose = pose.astype('float').reshape(-1,2)
    pose_x = frame_and_data.iloc[n,1]
    pose_y = frame_and_data.iloc[n,2]
    astar = frame_and_data.iloc[n,14:16].values
    astar = astar.astype('float').reshape(-1,2)
    astar_x = frame_and_data.iloc[n,14]
    astar_y = frame_and_data.iloc[n,15]

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
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    return(loss, optimizer)


def get_loader(dataset, batch_size, shuffle, num_workers, sampler):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    return loader


# def TrainNet(net, batch_size_train, batch_size_val, shuffle, num_workers, data, train_sampler, val_sampler, n_epochs, learning_rate):
#
#     # Print all of the hyperparameters of the training iteration:
#     print("===== HYPERPARAMETERS =====")
#     print("batch_size=", batch_size_train)
#     print("epochs=", n_epochs)
#     print("learning_rate=", learning_rate)
#     print("=" * 30)
#
#     # Get dataloaders
#     train_loader = get_loader(data, batch_size_train, shuffle, num_workers, sampler=train_sampler)
#     val_loader = get_loader(data, batch_size_val, shuffle=False, num_workers=2, sampler=val_sampler)
#
#     # Get training data
#     # train_loader = get_train_loader(batch_size, training_set, train_sampler)
#     n_batches = len(train_loader)
#
#     # Create our loss and optimizer functions
#     loss, optimizer = createLossAndOptimizer(net, learning_rate)
#
#     # Time for printing
#     training_start_time = time.time()
#
#     # Loop for n_epochs
#     for epoch in range(n_epochs):
#
#         running_loss = 0.0
#         print_every = n_batches / 10
#         start_time = time.time()
#         total_train_loss = 0
#
#         # train_loader.dataset.indices = train_loader.dataset.indices.tolist()
#         for i, data in enumerate(train_loader):
#
#             # Get inputs
#             # inputs, labels = data
#             img_input, pose_input, expert = data['image'], data['pose'], data['astar']
#
#             # Wrap them in a Variable object
#             # inputs, labels = Variable(inputs), Variable(labels)
#             img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)
#
#             if next(net.parameters()).is_cuda == True:
#                 img_input = img_input.to(torch.device("cuda"))
#                 pose_input = pose_input.to(torch.device("cuda"))
#                 expert = expert.to(torch.device("cuda"))
#
#             # Set the parameter gradients to zero
#             optimizer.zero_grad()
#
#             # Forward pass, backward pass, optimize
#             # outputs = net(pose_input[i], img_input[i])
#             outputs = net(pose_input, img_input)
#
#             # loss_size = loss(outputs, expert[i].float())
#             loss_size = loss(outputs, expert.float())
#
#             loss_size.backward()
#             optimizer.step()
#
#             # Print statistics
#             running_loss += loss_size.data
#             total_train_loss += loss_size.data
#
#             # Print every 10th batch of an epoch
#             # if (i + 1) % (print_every + 1) == 0:
#             print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
#                 epoch + 1, int(100 * (i + 1) / n_batches), running_loss / 1,
#                 time.time() - start_time))
#             # Reset running loss and time
#             running_loss = 0.0
#             start_time = time.time()
#
#         # At the end of the epoch, do a pass on the validation set
#         total_val_loss = 0
#         for i_val, data_val in enumerate(val_loader):
#             # Wrap tensors in Variables
#             img_input_val, pose_input_val, expert_val = Variable(data_val['image']), Variable(data_val['pose'])\
#                 , Variable(data_val['astar'])
#
#             if next(net.parameters()).is_cuda == True:
#                 img_input_val = img_input_val.to(torch.device("cuda"))
#                 pose_input_val = pose_input_val.to(torch.device("cuda"))
#                 expert_val = expert_val.to(torch.device("cuda"))
#
#             # Forward pass
#             val_outputs = net(pose_input_val, img_input_val)
#             val_loss_size = loss(val_outputs, expert_val.float())
#             total_val_loss += val_loss_size.data
#
#         print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
#
#     print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
#
#
#
# def TestNet(net, batch_size, data, sampler):
#
#     test_loader = get_loader(data, batch_size, shuffle=False, num_workers=2, sampler=sampler)
#
#     net.eval()
#     loss_func = torch.nn.MSELoss()
#     test_loss = 0
#     correct = 0
#
#     for j, data in enumerate(test_loader):
#         img_input, pose_input, expert = data['image'], data['pose'], data['astar']
#         # outputs = model(pose_input[j], img_input[j])
#
#         img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)
#
#         if next(net.parameters()).is_cuda == True:
#             img_input = img_input.to(torch.device("cuda"))
#             pose_input = pose_input.to(torch.device("cuda"))
#             expert = expert.to(torch.device("cuda"))
#
#         outputs = net(pose_input, img_input)
#
#         # test_loss += F.nll_loss(outputs, expert[j].float(), size_average=False).data[0]
#
#         test_loss += loss_func(outputs, expert.float())
#
#         if test_loss < 0.01:
#             correct = correct + 1
#
#         # pred = outputs.data.max(1,keepdim=True)[1]
#
#         # correct += pred.eq(expert[j].float().data.view_as(pred)).sum()
#
#         test_loss /= len(test_loader)
#
#         print("Model Output: {}".format(outputs.data) + " \n" + "Expert Output: {}".format(expert.data))
#
#         # test_loss /= len(test_loader.dataset)
#     print('Test set: Average loss: {:.4f}'.format(test_loss))


def TrainNet_new(net, batch_size_train, batch_size_val, shuffle, num_workers, train_data, val_data, n_epochs, learning_rate):

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size_train)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    train_sampler=SequentialSampler(train_data)
    val_sampler = SequentialSampler(val_data)

    # Get dataloaders
    train_loader = get_loader(train_data, batch_size_train, shuffle=shuffle,
                              num_workers=num_workers, sampler=train_sampler)
    val_loader = get_loader(val_data, batch_size_val, shuffle=False, num_workers=2,
                            sampler=val_sampler)

    # Get training data
    # train_loader = get_train_loader(batch_size, training_set, train_sampler)
    n_batches = len(train_loader)

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

        # train_loader.dataset.indices = train_loader.dataset.indices.tolist()
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

            # loss_size = loss(outputs, expert[i].float())
            loss_size = loss(outputs, expert.squeeze().float())

            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            # Print every 10th batch of an epoch
            # if (i + 1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.4f} took: {:.2f}s".format(
                epoch + 1, int(100 * (i + 1) / n_batches), running_loss / 1,
                time.time() - start_time))
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

            # plotter.plot('loss', 'train', 'Class Loss', epoch, total_train_loss / (epoch+1))
            # plotter.plot('loss', 'val', 'Class Loss', epoch, total_val_loss / len(val_loader))

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for i_val, data_val in enumerate(val_loader):
            # Wrap tensors in Variables
            img_input_val, pose_input_val, expert_val = Variable(data_val['image']), Variable(data_val['pose']), Variable(data_val['astar'])

            if next(net.parameters()).is_cuda == True:
                img_input_val = img_input_val.to(torch.device("cuda"))
                pose_input_val = pose_input_val.to(torch.device("cuda"))
                expert_val = expert_val.to(torch.device("cuda"))

            # Forward pass
            val_outputs = net(pose_input_val, img_input_val)
            val_loss_size = loss(val_outputs, expert_val.squeeze().float())
            total_val_loss += val_loss_size.data

        print("Validation loss = {:.4f}".format(total_val_loss / len(val_loader)))

        # Save current model in tmp pt file
        if epoch % 5 == 0:
            label = epoch
            net_name = 'Models/dnn_4-23_v5' + '_' + str(label) + '.pt'
            torch.save(net.state_dict(), net_name)

        plotter.plot('loss', 'train', 'Class Loss', epoch, total_train_loss / len(train_loader))
        plotter.plot('loss', 'val', 'Class Loss', epoch, total_val_loss / len(val_loader))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))



def TestNet_new(net, batch_size, data):

    test_sampler = SequentialSampler(data)

    test_loader = get_loader(data, batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    net.eval()
    loss_func = torch.nn.MSELoss()
    test_loss = 0
    correct = 0

    for j, data in enumerate(test_loader):
        img_input, pose_input, expert = data['image'], data['pose'], data['astar']
        # outputs = model(pose_input[j], img_input[j])


        viz = False

        if viz:
            fig = plt.figure()
            plt.title('Transformed Image (64x64)')
            # show_img_batch(test_loader)
            grid = utils.make_grid(img_input[40])
            plt.imshow(grid.numpy().transpose((1, 2, 0)))




        img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)

        if next(net.parameters()).is_cuda == True:
            img_input = img_input.to(torch.device("cuda"))
            pose_input = pose_input.to(torch.device("cuda"))
            expert = expert.to(torch.device("cuda"))



        # plt.title('Batch from dataloader')

        outputs = net(pose_input, img_input)
        _, preds = torch.max(outputs, 1)
        expert = expert.squeeze()

        test_loss += loss_func(outputs, expert.float())

        # if test_loss < 0.01:
        #     correct = correct + 1

        # correct += preds.eq(expert[j].float().data.view_as(preds)).sum()
        # correct += preds.eq(expert).data.view_as(preds).sum()

        test_loss /= len(test_loader)

        print("Model Prediction: {}".format(outputs) + " \n" + "Expert Output: {}".format(expert))

        # test_loss /= len(test_loader.dataset)
        # print('Test set: Average loss: {:.4f}'.format(test_loss))


    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))


def main():
    # Initialize model
    model = DNN_C4F4F2_Astar_small()
    model = DNN_C4F4F2_Astar()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model.to(device)
    # model.to("cpu")

    # define parameters
    batch_size_train = 64
    batch_size_val = 64
    batch_size_test = 64

    # Define transforms
    # transformations = transforms.Compose([Crop((360,640)), Rescale(128), Grayscale(), ToTensor_Norm()])
    # transformations = transforms.Compose([Crop((360,640)), Rescale(64), Grayscale(), ToTensor()])
    # transformations = transforms.Compose([Crop((360,640)), Rescale(128), ToTensor_Norm()])
    transformations = transforms.Compose([Crop((360, 640)), Rescale(64), Grayscale(), ToTensor()])

    # Prepare Visdom
    global plotter
    plotter = VisdomLinePlotter(env_name='Obstacle Avoidance DNN')

    # Datasets
    training_data = ObstacleAvoidanceDataset_new(
        csv_file='/home/charris/Desktop/dataset_4-23/dataset_4-23_training.csv',
        root_dir='/home/charris/Desktop/dataset_4-23', transform=transformations)

    testing_data = ObstacleAvoidanceDataset_new(
        csv_file='/home/charris/Desktop/dataset_4-23/dataset_4-23_testing.csv',
        root_dir='/home/charris/Desktop/dataset_4-23', transform=transformations)
    validation_data = ObstacleAvoidanceDataset_new(
        csv_file='/home/charris/Desktop/dataset_4-23/dataset_4-23_validation.csv',
        root_dir='/home/charris/Desktop/dataset_4-23', transform=transformations)

    print("Beginning Training...")
    TrainNet_new(model, batch_size_train, batch_size_val, shuffle=False, num_workers=2, train_data=training_data,
                 val_data=validation_data, n_epochs=100, learning_rate=0.005)
    print("Saving model...")
    torch.save(model.state_dict(), 'Models/dnn_4-23_v5.pt')

    print("Loading model...")
    model.to("cpu")
    print("cpu")

    # with torch.no_grad()
    model.load_state_dict(torch.load('Models/dnn_4-23_v5_55.pt'))
    print("Beginning Testing...")
    TestNet_new(model, batch_size_test, data=testing_data)


if __name__ == "__main__":
    main()
    # datasetadvanced()


