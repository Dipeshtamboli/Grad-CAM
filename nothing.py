import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
import matplotlib.pyplot as plt  
from sklearn.metrics import f1_score, classification_report, confusion_matrix
writer = SummaryWriter()

transform = transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor(),])
data_train = torchvision.datasets.ImageFolder("/home/nim/grad_cam/data/train", transform=transform)
data_test = torchvision.datasets.ImageFolder("/home/nim/grad_cam/data/test", transform=transform)


class Net(nn.Module):
    def _init_(self):
        super(Net, self)._init_()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.LeakyReLU(0.01)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4 = nn.LeakyReLU(0.01)
        self.dropout1=nn.Dropout(p=0.5, inplace=False)
        self.dropout2=nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(4*4*128,128)
        self.fc2 = nn.Linear(128, 21)
        self.relu5 = nn.LeakyReLU(0.01)
        # self.fc3 = nn.Linear(200, 100)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = self.relu3(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu4(self.conv4(x))

        # mayank - HIGH hai kya?? 3 baar max pool?? Ek baar kar le
        x = F.max_pool2d(x, 2, 2)
        x = F.max_pool2d(x, 2, 2)
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = x.view(-1, 4*4*128)
        x = self.relu5(self.fc1(x))
        # x = m(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
#         self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
#         self.dropout1=nn.Dropout(p=0.5, inplace=False)
#         self.dropout2=nn.Dropout(p=0.5, inplace=False)
#         self.fc1 = nn.Linear(4*4*128,128)
#         self.fc2 = nn.Linear(128, 21)
#         # self.fc3 = nn.Linear(200, 100)

#     def forward(self, x):
#         # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
#         m=nn.LeakyReLU(0.01)
#         # print(x.shape)
#         x = m(self.conv1(x))
#         # print(x.shape)
#         x = F.max_pool2d(x, 2, 2)
#         x = m(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)

#         x = m(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = m(self.conv4(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.max_pool2d(x, 2, 2)
#         x = F.max_pool2d(x, 2, 2)
#         x = self.dropout1(x)
#         x = x.view(-1, 4*4*128)
#         x = m(self.fc1(x))
#         # x = m(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)

#         return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTrain_accuracy: {:.0f}%\n'.format(100. * correct / len(train_loader.dataset)))
    writer.add_scalar('train_Accuracy_epoch',100. * correct / len(train_loader.dataset),epoch)
    writer.add_scalar('train_loss_epoch',loss/len(train_loader.dataset),epoch)

def test(args, model, device, test_loader,epoch):
    print("test started")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('test_loss_epoch',test_loss,epoch)
    writer.add_scalar('test_Accuracy_epoch',100. * correct / len(test_loader.dataset),epoch)

def main():
    start = time.time()
    print ("into the main")
    parser = argparse.ArgumentParser(description='UC_Merced data_mynet')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    

    parser.add_argument('--lr', type=float, default=0.0003   , metavar='LR',
                        help='learning rate (default: 0.01)')
   
    parser.add_argument('--momentum', type=float, default=0.4, metavar='M',
                        help='SGD momentum (default: 0.9)')
   
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
   
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
   
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
#    device = "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=32 ,shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=32 ,shuffle = False, **kwargs)
    print("device: ",device)
    
    model = Net().to(device)
    print ("model transferred to device")
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=args.momentum, centered=False)
    print ("optimizer choosed")
    print("#######__parameters__######")
    print("learning rate: ", args.lr, "\nmomentum: ", args.momentum, "\nepochs: ", args.epochs)
    print("############################")    
    print("model:\n",model)
    print("############################")
    # print("optimizer:\n",optimizer)
    print("############################")

    # for epoch in range(2):
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader, epoch)
        # writer.add_scalar('loss_fn2',loss.item(),epoch)
    if (args.save_model):
        torch.save(model.state_dict(),"/home/nim/grad_cam/pt_files/file: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" CIFAR_100_cnn.pt")
	# /home/nim/grad_cam/models
    save_name_pkl = "/home/nim/grad_cam/pickel_files/file: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.pkl"
    save_name_txt = "/home/nim/grad_cam/models/file: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.txt"
    model_file = open(save_name_txt,"w") 
    model_string = str(model)
    optimizer_string = str(optimizer)
    model_file.write(model_string)
    model_file.write(optimizer_string)
    model_file.write(save_name_txt)
    model_file.close()
   
    f=open(save_name_pkl,"wb")
    pickle.dump(model, f)
    end = time.time()
    print('time taken is ', (end-start))


if __name__ == '__main__':
    main()
