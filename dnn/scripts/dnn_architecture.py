import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class DNN_C4F4F2(nn.Module):

    def __init__(self):
        super(DNN_C4F4F2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 500)

        self.fc3 = nn.Linear(502, 100)
        self.fc4 = nn.Linear(100, 20)
        self.fc5 = nn.Linear(20,2)

    def forward(self, pose_input, img_input):
        cnnout = self.layer1(img_input.float())
        cnnout = self.layer2(cnnout)
        cnnout = self.layer3(cnnout)
        cnnout = self.layer4(cnnout)
        cnnout = cnnout.view(cnnout.size(0), -1)
        cnnout = F.relu(self.fc1(cnnout))
        cnnout = self.fc2(cnnout)
        cnnout = F.relu(cnnout)

        posein = F.relu(pose_input.float().squeeze())

        # pose input is [x,y]
        dnn_input = torch.cat((posein, cnnout), dim=1)

        out = F.relu(self.fc3(dnn_input))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)

        return out


class DNN_C4F2F3_Gray(nn.Module):

    def __init__(self):
        super(DNN_C4F2F3_Gray, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(41472, 5000)
        self.fc2 = nn.Linear(5000, 500)

        self.fc3 = nn.Linear(502, 100)
        self.fc4 = nn.Linear(100, 20)
        self.fc5 = nn.Linear(20,2)

    def forward(self, pose_input, img_input):
        cnnout = self.layer1(img_input.float())
        cnnout = self.layer2(cnnout)
        cnnout = self.layer3(cnnout)
        cnnout = self.layer4(cnnout)
        cnnout = cnnout.view(cnnout.size(0), -1)
        cnnout = F.relu(self.fc1(cnnout))
        cnnout = self.fc2(cnnout)

        cnnout = F.relu(cnnout)

        posein = F.relu(pose_input.float().squeeze())

        # pose input is [x,y]
        dnn_input = torch.cat((posein, cnnout), dim=1)

        out = F.relu(self.fc3(dnn_input))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)

        return out


class DNN_Classifier_small(nn.Module):

    def __init__(self):
        super(DNN_Classifier_small, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 56)

        self.fc3 = nn.Linear(58, 28)
        self.fc4 = nn.Linear(28, 9)

    def forward(self, pose_input, img_input):
        cnnout = self.layer1(img_input.float())
        cnnout = self.layer2(cnnout)
        cnnout = self.layer3(cnnout)
        cnnout = self.layer4(cnnout)
        cnnout = cnnout.view(cnnout.size(0), -1)
        cnnout = F.relu(self.fc1(cnnout))
        cnnout = self.fc2(cnnout)
        cnnout = F.relu(cnnout)

        posein = F.relu(pose_input.float().squeeze())

        # pose input is [x,y]
        if posein.dim() == 1:
            posein = posein.reshape((1, 2))
        dnn_input = torch.cat((posein, cnnout), dim=1)

        out = F.relu(self.fc3(dnn_input))
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)



class DNN_C4F4F2_Astar(nn.Module):

    def __init__(self):
        super(DNN_C4F4F2_Astar, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(20736, 5096)
        self.fc2 = nn.Linear(5096, 128)

        self.fc3 = nn.Linear(130, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, pose_input, img_input):
        cnnout = self.layer1(img_input.float())
        cnnout = self.layer2(cnnout)
        cnnout = self.layer3(cnnout)
        cnnout = self.layer4(cnnout)
        cnnout = cnnout.view(cnnout.size(0), -1)
        cnnout = F.relu(self.fc1(cnnout))
        cnnout = self.fc2(cnnout)
        cnnout = F.relu(cnnout)

        posein = pose_input.float()
        posein = posein.reshape((posein.shape[0],posein.shape[2]))
        posein = F.relu(posein)

        # pose input is [x,y]
        dnn_input = torch.cat((posein, cnnout), dim=1)

        out = F.relu(self.fc3(dnn_input))
        out = self.fc4(out)

        return out


class DNN_C4F4F2_Astar_small(nn.Module):

    def __init__(self):
        super(DNN_C4F4F2_Astar_small, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 36)

        self.fc3 = nn.Linear(38, 12)
        self.fc4 = nn.Linear(12, 2)

    def forward(self, pose_input, img_input, viz = False):
        layer1 = self.layer1(img_input.float())
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if viz:
            fig = plt.figure()
            layer1_graph = layer1[40].data
            for idx, filt in enumerate(layer1_graph):
                plt.subplot(2, 16, idx + 1)
                plt.imshow(filt, cmap="gray")
                plt.axis('off')

            fig2 = plt.figure()
            layer2_graph = layer2[40].data
            for idx, filt in enumerate(layer2_graph):
                plt.subplot(4, 16, idx + 1)
                plt.imshow(filt, cmap="gray")
                plt.axis('off')

            fig3 = plt.figure()
            layer3_graph = layer3[40].data
            for idx, filt in enumerate(layer3_graph):
                plt.subplot(8, 16, idx + 1)
                plt.imshow(filt, cmap="gray")
                plt.axis('off')

            fig4 = plt.figure()
            layer4_graph = layer4[40].data
            for idx, filt in enumerate(layer4_graph):
                plt.subplot(8, 32, idx + 1)
                plt.imshow(filt, cmap="gray")
                plt.axis('off')

        cnnout = layer4.view(layer4.size(0), -1)
        cnnout = F.relu(self.fc1(cnnout))
        cnnout = F.relu(cnnout)

        posein = pose_input.float()
        posein = posein.reshape((posein.shape[0],posein.shape[2]))
        posein = F.relu(posein)

        # pose input is [x,y]
        dnn_input = torch.cat((posein, cnnout), dim=1)

        out = F.relu(self.fc3(dnn_input))
        out = self.fc4(out)

        return out


# class DNN_C4F2F3(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F2F3, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=2),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(41472, 5000)
#         self.fc2 = nn.Linear(5000, 500)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(502, 100)
#         self.fc4 = nn.Linear(100, 20)
#         self.fc5 = nn.Linear(20,2)
#
#         # self.fc4 = nn.Linear(38,8)
#         # self.fc5 = nn.Linear(8,2)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         # pose input is [x,y]
#         # dnn_input = torch.cat((pose_input.float().squeeze(), cnnout), dim=1)
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = F.relu(self.fc4(out))
#         out = self.fc5(out)
#
#         return out
#
#
#
# class DNN_C4F2F3_Gray(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F2F3_Gray, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=2),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(41472, 5000)
#         self.fc2 = nn.Linear(5000, 500)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(502, 100)
#         self.fc4 = nn.Linear(100, 20)
#         self.fc5 = nn.Linear(20,2)
#
#         # self.fc4 = nn.Linear(38,8)
#         # self.fc5 = nn.Linear(8,2)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         # pose input is [x,y]
#         # dnn_input = torch.cat((pose_input.float().squeeze(), cnnout), dim=1)
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = F.relu(self.fc4(out))
#         out = self.fc5(out)
#
#         return out
#
#
#
#
# class DNN_GT(nn.Module):
#
#     def __init__(self):
#         super(DNN_GT, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         self.fc1 = nn.Linear(46208, 512)
#         self.fc2 = nn.Linear(512, 36)
#
#         self.fc3 = nn.Linear(38, 8)
#         self.fc4 = nn.Linear(8, 2)
#
#
#     def forward(self, pose_input, img_input):
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = self.layer5(cnnout)
#         cnnout = self.layer6(cnnout)
#
#         cnnout = cnnout.view(cnnout.size(0), -1)
#
#         cnnout = self.fc1(cnnout)
#         cnnout = F.relu(self.fc2(cnnout))
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return out
#
#
# class DNN_GT(nn.Module):
#
#     def __init__(self):
#         super(DNN_GT, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         self.fc1 = nn.Linear(46208, 512)
#         self.fc2 = nn.Linear(512, 36)
#
#         self.fc3 = nn.Linear(38, 8)
#         self.fc4 = nn.Linear(8, 2)
#
#
#     def forward(self, pose_input, img_input):
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = self.layer5(cnnout)
#         cnnout = self.layer6(cnnout)
#
#         cnnout = cnnout.view(cnnout.size(0), -1)
#
#         cnnout = self.fc1(cnnout)
#         cnnout = F.relu(self.fc2(cnnout))
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return out
#
#
# class DNN_C4F4F2_Classifier(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F4F2_Classifier, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1,32,kernel_size=3,padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(6400, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(130, 64)
#         self.fc4 = nn.Linear(64, 9)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         # pose input is [x,y]
#         # dnn_input = torch.cat((pose_input.float().squeeze(), cnnout), dim=1)
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return F.log_softmax(out,dim=1)
#
#
# class DNN_C4F4F2_Classifier2(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F4F2_Classifier2, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(20736, 5096)
#         self.fc2 = nn.Linear(5096, 128)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(130, 64)
#         self.fc4 = nn.Linear(64, 9)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = F.relu(pose_input.float().squeeze())
#
#         # pose input is [x,y]
#         # dnn_input = torch.cat((pose_input.float().squeeze(), cnnout), dim=1)
#         posein = posein.reshape((1,2))
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return F.log_softmax(out,dim=1)
#
#
#
# class DNN_GT_MPC(nn.Module):
#
#     def __init__(self):
#         super(DNN_GT_MPC, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         self.fc1 = nn.Linear(46208, 512)
#         self.fc2 = nn.Linear(512, 36)
#
#         self.fc3 = nn.Linear(38, 8)
#         self.fc4 = nn.Linear(8, 4)
#
#
#     def forward(self, pose_input, img_input):
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = self.layer5(cnnout)
#         cnnout = self.layer6(cnnout)
#
#         cnnout = cnnout.view(cnnout.size(0), -1)
#
#         cnnout = self.fc1(cnnout)
#         cnnout = F.relu(self.fc2(cnnout))
#
#         posein = pose_input.float()
#         posein = posein.reshape((posein.shape[0],posein.shape[2]))
#         posein = F.relu(posein)
#         # posein = F.relu(pose_input.float().squeeze())
#
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return out
#
#
# class DNN_C4F4F2_MPC(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F4F2_MPC, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(20736, 5096)
#         self.fc2 = nn.Linear(5096, 128)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(141, 64)
#         self.fc4 = nn.Linear(64, 4)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = pose_input.float()
#         posein = posein.reshape((posein.shape[0],posein.shape[2]))
#         posein = F.relu(posein)
#
#         # pose input is [x,y]
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return out

#
# class DNN_C4F4F2_mpc_small(nn.Module):
#
#     def __init__(self):
#         super(DNN_C4F4F2_mpc_small, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1,32,kernel_size=3,padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#
#         # self.fc1 = nn.Linear(2304, 1000)
#         self.fc1 = nn.Linear(6400, 512)
#         self.fc2 = nn.Linear(512, 36)
#         # self.fc3 = nn.Linear(500, 36)
#
#         self.fc3 = nn.Linear(38, 12)
#         self.fc4 = nn.Linear(12, 4)
#
#     def forward(self, pose_input, img_input):
#         # cnnout = self.layer1(img_input.float()[None])
#         cnnout = self.layer1(img_input.float())
#         cnnout = self.layer2(cnnout)
#         cnnout = self.layer3(cnnout)
#         cnnout = self.layer4(cnnout)
#         cnnout = cnnout.view(cnnout.size(0), -1)
#         cnnout = F.relu(self.fc1(cnnout))
#         cnnout = self.fc2(cnnout)
#         # cnnout = F.relu(self.fc2(cnnout))
#         # cnnout = self.fc3(cnnout)
#         # cnnout = F.relu(cnnout, dim=1)
#         cnnout = F.relu(cnnout)
#
#         posein = pose_input.float()
#         posein = posein.reshape((posein.shape[0],posein.shape[2]))
#         posein = F.relu(posein)
#
#         # pose input is [x,y]
#         dnn_input = torch.cat((posein, cnnout), dim=1)
#
#         # dnn_input = torch.stack(tensor_array)
#
#         # # out = F.relu(self.fc4(dnn_input))
#         # out = self.fc4(dnn_input)
#         #
#         # out = self.fc5(out)
#
#         out = F.relu(self.fc3(dnn_input))
#         out = self.fc4(out)
#
#         return out
