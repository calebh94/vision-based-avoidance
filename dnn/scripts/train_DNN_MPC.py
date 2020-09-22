# from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
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
from avoidance_framework import DNN_GT_MPC, DNN_C4F4F2_MPC, DNN_C4F4F2_mpc_small


class ObstacleAvoidanceDataset_MPC(Dataset):
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
        traj = self.data.iloc[idx,0]
        img_name = self.data.iloc[idx,1]
        img_name_full = os.path.join(self.root_dir, img_name)
        image = io.imread(img_name_full)
        pose = self.data.iloc[idx, 2:15].values
        pose = pose.astype('float').reshape(-1, 13)
        mpc = self.data.iloc[idx, 18:22].values
        mpc = mpc.astype('float').reshape(-1, 4)
        sample = {'image': image, 'pose': pose, 'mpc': mpc, 'batch': traj}

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



# class ObstacleAvoidanceSampler(Sampler):
#     """ OBstacle Avoidance special Problem Sampler for Pytorch"""
#
#     def __init__(self):


class Grayscale(object):

    def __call__(self, sample):
        img, pose, mpc, batch = sample['image'], sample['pose'], sample['mpc'], sample['batch']
        img = rgb2gray(img)
        img = np.expand_dims(img, 2)

        return {'image': img, 'pose': pose, 'mpc': mpc, 'batch': batch}



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
        image, pose, mpc, batch = sample['image'], sample['pose'], sample['mpc'], sample['batch']

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

        return {'image': img, 'pose': pose, 'mpc': mpc, 'batch': batch}


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
        image, pose, mpc, batch = sample['image'], sample['pose'], sample['mpc'], sample['batch']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # top = np.random.randint(0, h - new_h)
        top = h - new_h
        # left = np.random.randint(0, w - new_w)
        left = w - new_w

        img = image[top: top + new_h,
                      left: left + new_w]

        return {'image': img, 'pose': pose, 'mpc': mpc, 'batch': batch}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose, mpc, batch = sample['image'], sample['pose'], sample['mpc'], sample['batch']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'pose': torch.from_numpy(pose),
                'mpc': torch.from_numpy(mpc),
                'batch': batch}

class ToTensor_Norm(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose, mpc, batch = sample['image'], sample['pose'], sample['mpc'], sample['batch']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': F.normalize(torch.from_numpy(image),3),
                'pose': F.normalize(torch.from_numpy(pose)),
                'mpc': torch.from_numpy(mpc),
                'batch': batch}


def show_img_batch(sample_batched):
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


def datasetadvanced():
    transformed_dataset = ObstacleAvoidanceDataset_MPC(
        csv_file='/media/charris/DISCOVERY/datasets/2019-04-10-21-36-41/2019-04-10-21-36-41.csv',
        root_dir='/media/charris/DISCOVERY/datasets/2019-04-10-21-36-41/',
                                                   transform=transforms.Compose([
                                                       Crop((360,640)),
                                                       Rescale(128),
                                                       # Grayscale(),
                                                       ToTensor()
                                                   ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['pose'].size(), sample['mpc'].size(), str(sample['batch']))

        if i==3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True,num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['pose'].size(),
              sample_batched['mpc'].size(), str(sample_batched['batch']))

        if i_batch==3:
            plt.figure()
            show_img_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


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


def TrainNet(net, batch_size_train, batch_size_val, shuffle, num_workers, train_data, val_data, n_epochs, learning_rate):

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
            img_input, pose_input, expert = data['image'], data['pose'], data['mpc']

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
            loss_size = loss(outputs, expert.float())

            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            # Print every 10th batch of an epoch
            # if (i + 1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.6f} took: {:.2f}s".format(
                epoch + 1, int(100 * (i + 1) / n_batches), running_loss / 1,
                time.time() - start_time))
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for i_val, data_val in enumerate(val_loader):
            # Wrap tensors in Variables
            img_input_val, pose_input_val, expert_val = Variable(data_val['image']), Variable(data_val['pose'])\
                , Variable(data_val['mpc'])

            if next(net.parameters()).is_cuda == True:
                img_input_val = img_input_val.to(torch.device("cuda"))
                pose_input_val = pose_input_val.to(torch.device("cuda"))
                expert_val = expert_val.to(torch.device("cuda"))

            # Forward pass
            val_outputs = net(pose_input_val, img_input_val)
            val_loss_size = loss(val_outputs, expert_val.float())
            total_val_loss += val_loss_size.data

        print("Validation loss = {:.6f}".format(total_val_loss / len(val_loader)))

        # Save current model in tmp pt file
        if epoch % 5 == 0:
            net_name = '/media/charris/57FB-AE20/Models/mpc_4-23/dnn_mpc_4-23_v2' + '_' + str(epoch) + '.pt'
            torch.save(net.state_dict(), net_name)

        plotter.plot('loss', 'train', 'Class Loss', epoch, total_train_loss / len(train_loader))
        plotter.plot('loss', 'val', 'Class Loss', epoch, total_val_loss / len(val_loader))

    print("Training finished, took {:.6f}s".format(time.time() - training_start_time))



def TestNet(net, batch_size, data):

    test_sampler = SequentialSampler(data)

    test_loader = get_loader(data, batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    net.eval()
    loss_func = torch.nn.MSELoss()
    test_loss = 0
    correct = 0

    for j, data in enumerate(test_loader):
        img_input, pose_input, expert = data['image'], data['pose'], data['mpc']
        # outputs = model(pose_input[j], img_input[j])

        img_input, pose_input, expert = Variable(img_input), Variable(pose_input), Variable(expert)

        if next(net.parameters()).is_cuda == True:
            img_input = img_input.to(torch.device("cuda"))
            pose_input = pose_input.to(torch.device("cuda"))
            expert = expert.to(torch.device("cuda"))

        outputs = net(pose_input, img_input)

        # test_loss += F.nll_loss(outputs, expert[j].float(), size_average=False).data[0]

        test_loss += loss_func(outputs, expert.float())

        if test_loss < 0.01:
            correct = correct + 1

        # pred = outputs.data.max(1,keepdim=True)[1]

        # correct += pred.eq(expert[j].float().data.view_as(pred)).sum()

        test_loss /= len(test_loader)

        print("Model Output: {}".format(outputs.data) + " \n" + "Expert Output: {}".format(expert.data))

        # test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}'.format(test_loss))


def main():
    # Initialize model
    model = DNN_C4F4F2_MPC()
    # model = DNN_C4F4F2_mpc_small()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # model.to(device)
    model.to("cpu")

    # Define transforms
    transformations = transforms.Compose([Crop((360,640)), Rescale(128), ToTensor()])
    # transformations = transforms.Compose([Crop((360,640)), Rescale(64), Grayscale(), ToTensor()])



    # Datasets
    training_data = ObstacleAvoidanceDataset_MPC(
        csv_file='/home/charris/Desktop/datasets/dataset_4-15_v1/dataset_4-15_training.csv',
        root_dir='/home/charris/Desktop/datasets/dataset_4-15_v1', transform=transformations)

    testing_data = ObstacleAvoidanceDataset_MPC(
        csv_file='/home/charris/Desktop/datasets/dataset_4-15_v1/dataset_4-15_testing.csv',
        root_dir='/home/charris/Desktop/datasets/dataset_4-15_v1', transform=transformations)
    validation_data = ObstacleAvoidanceDataset_MPC(
        csv_file='/home/charris/Desktop/datasets/dataset_4-15_v1/dataset_4-15_validation.csv',
        root_dir='/home/charris/Desktop/datasets/dataset_4-15_v1', transform=transformations)

    # define parameters
    batch_size_train = 256
    batch_size_val = 128
    batch_size_test = 128

    global plotter
    plotter = VisdomLinePlotter(env_name='Obstacle Avoidance DNN')


    # load = False
    # if load:
    #     print("Loading pre-trained model...")
    #     model.load_state_dict(torch.load('Models/dnn_4-8_v1.pt'))

    print("Beginning Training...")
    TrainNet(model, batch_size_train, batch_size_val, shuffle=False, num_workers=2, train_data=training_data,
             val_data=validation_data, n_epochs=100, learning_rate=0.01)

    print("Saving model...")
    torch.save(model.state_dict(), 'Models/dnn_mpc_4-23_v2.pt')

    print("Loading model...")
    model.to("cpu")

    # with torch.no_grad()
    model.load_state_dict(torch.load('/media/charris/57FB-AE20/Models/mpc_4-23/dnn_mpc_4-23_v2_20.pt'))
    print("Beginning Testing...")
    TestNet(model, batch_size_test, data=testing_data)


if __name__ == "__main__":
    main()
    # datasetadvanced()


