from __future__ import print_function
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



transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),])
data_train = torchvision.datasets.ImageFolder("/home/dipesh/grad_cam/data/train", transform=transform)
data_test = torchvision.datasets.ImageFolder("/home/dipesh/grad_cam/data/test", transform=transform)

# print (train_data).type

vgg_based = torchvision.models.vgg19(pretrained=True)
for param in vgg_based.parameters():
   param.requires_grad = False

# Modify the last layer
# number_features = vgg_based.classifier[6].in_features
features = list(vgg_based.classifier.children())[:-1] # Remove last layer
# print(type(vgg_based.classifier.children()))

features.extend([torch.nn.Linear(4096, 21)])
vgg_based.classifier = torch.nn.Sequential(*features)


# print(vgg_based)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
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
    return 100. * correct / len(train_loader.dataset)

def test(args, model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('test_loss_epoch',test_loss,epoch)
    writer.add_scalar('test_Accuracy_epoch',100. * correct / len(test_loader.dataset),epoch)
    return 100. * correct / len(test_loader.dataset)

def main():
    start = time.time()
    
    parser = argparse.ArgumentParser(description='UC-Mercer satellite dataset')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    

    parser.add_argument('--lr', type=float, default=0.0003   , metavar='LR',
                        help='learning rate (default: 0.01)')
   
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
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
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=32 ,shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=32 ,shuffle = False, **kwargs)
    print("device: ",device)
    model = vgg_based.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=args.momentum, centered=False)

    print("#######__parameters__######")
    print("learning rate: ", args.lr, "\nmomentum: ", args.momentum, "\nepochs: ", args.epochs)
    print("############################")    
    print(type(model.state_dict()))
    print("model:\n",model)
    print("############################")
    print("optimizer:\n",optimizer)
    print("############################")

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader, epoch)
   
    if (args.save_model):
        torch.save(model.state_dict(),"/home/nim/grad_cam/pt_files/file:pretrained: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" CIFAR_100_cnn.pt")
    # /home/nim/grad_cam/models
    save_name_pkl = "/home/nim/grad_cam/pickel_files/file:pretrained: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.pkl"
    save_name_txt = "/home/nim/grad_cam/models/file:pretrained: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.txt"
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
