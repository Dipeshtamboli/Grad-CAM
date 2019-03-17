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
# 15886, 400 -> useful_patches
# 47070, 1100 -> useless_patches
# 62956, 1500 -> total data  

transform = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),])
# /home/dipesh/lab_work/data/useful_patches
train_data = torchvision.datasets.ImageFolder("/home/dipesh/lab_work/data/train", transform=transform)
test_data = torchvision.datasets.ImageFolder("/home/dipesh/lab_work/data/test", transform=transform)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, sampler=None, batch_sampler=None, num_workers=0)
# print(len(train_data))
# print(len(test_data))

# img, lab = train_data.__getitem__(0)
# print(img.shape)

# train_iter = iter(X_train)
# images , labels = train_iter.next()

########## Shape of data ##################
# print('images shape on batch size = {}'.format(images.size()))
# print('labels shape on batch size = {}'.format(labels.size()))
# print(images.size())
# print(labels.size())

# ###########Visualizing the images in the batch using matplotlib #############
# grid = torchvision.utils.make_grid(images)
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.axis('off')
# plt.title(labels.numpy());
# plt.show()

# ###########Visualizing the images in the batch using tensorboardx #############
# for n_iter in range(64):
#     x = images #torchvision.rand(32, 3, 64, 64)  # output from network
#     x = torchvision.utils.make_grid(x, normalize=True, scale_each=True)
#     writer.add_image('Image', x, n_iter)  # Tensor
vgg_based = torchvision.models.vgg19(pretrained=True)
for param in vgg_based.parameters():
   param.requires_grad = False

# Modify the last layer
# number_features = vgg_based.classifier[6].in_features
features = list(vgg_based.classifier.children())[:-1] # Remove last layer
# print(type(vgg_based.classifier.children()))

features.extend([torch.nn.Linear(4096, 2)])
vgg_based.classifier = torch.nn.Sequential(*features)


print(vgg_based)


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


            # writer.add_scalar('loss_fn',loss.item(),batch_idx * len(data)/10000)
            # writer.add_histogram("epoch: "+str(epoch),loss.item(),batch_idx * len(data)/10000)
            # writer.add_pr_curve('loss_fn',epoch,batch_idx * len(data),loss.item())
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # writer.add_scalar('train_Accuracy_iter',100. * correct / len(train_loader.dataset),batch_idx)
            writer.add_scalar('train_loss_iter',loss.item(),batch_idx)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target2=target.reshape(len(target),1)
        # print(classification_report(target2 , pred, target_names=None))
        # print("\nconfusion_matrix",confusion_matrix(target2,pred))
        # target2=target.reshape(len(target),1)
        # print("\ntar:",target2.size(),"pred\n",pred.size(),"\n\n\n")
        # print(len(target))
        # print("\n\n\n\n\n\n\noutput\n\n\n\n\n\n\n",pred,"tar\n",target2)

    print('\nTrain_accuracy: {:.0f}%\n'.format(100. * correct / len(train_loader.dataset)))
    writer.add_scalar('train_Accuracy_epoch',100. * correct / len(train_loader.dataset),epoch)
    writer.add_scalar('train_loss_epoch',loss/len(train_loader.dataset),epoch)
    return 100. * correct / len(train_loader.dataset)




def test(args, model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    sum_confusion_matrix=0;
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    		# target2=target.reshape(len(target),1)
            sum_confusion_matrix += confusion_matrix(target,pred)

    test_loss /= len(test_loader.dataset)

    # print("target\n",target,"\npred\n",pred)
    print(classification_report(target , pred, target_names=None))
    print("\nconfusion_matrix\n",sum_confusion_matrix)
    print(sum_confusion_matrix[0,0],sum_confusion_matrix[0,1],sum_confusion_matrix[1,0])
    precision = sum_confusion_matrix[0,0] /(sum_confusion_matrix[0,0]+sum_confusion_matrix[1,0])
    recall = sum_confusion_matrix[0,0] /(sum_confusion_matrix[0,0]+sum_confusion_matrix[0,1])
    print(recall,precision)
    f1_score_calculated = 2*precision*recall/(precision + recall)
    print("f1_score_calculated = ",f1_score_calculated)
    # target_names = ['class 0', 'class 1']
    # f1_result_micro = f1_score(target2,pred,average='micro')
    # print("\nf1_score:" ,f1_result_micro )
    # f1_result_binary = f1_score(target2,pred,average='binary')
    # print("\nf1_score:" ,f1_result_binary )
    # f1_result_weighted = f1_score(target2,pred,average='weighted')
    # print("\nf1_score:" ,f1_result_weighted )
    # f1_result_none = f1_score(target2,pred,average=None)
    # print("\nf1_score:" ,f1_result_none )


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('test_loss_epoch',test_loss,epoch)
    writer.add_scalar('test_Accuracy_epoch',100. * correct / len(test_loader.dataset),epoch)
    return 100. * correct / len(test_loader.dataset)




def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Example')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, sampler=None, batch_sampler=None, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128 ,shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128 ,shuffle = True, **kwargs)
    print("device: ",device)
    # model = Net().to(device)
    model = vgg_based.to(device)    
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=args.momentum, centered=False)

    print("#######__parameters__######")
    print("############################")    
    print(type(model.state_dict()))
    print("\nmodel:\n",model)
    print("############################")
    print("optimizer:\n",optimizer)
    print("\nepochs: ", args.epochs)
    print("############################")

    # print("string_model",a)

    # for epoch in range(2):
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader, epoch)
        # writer.add_scalar('loss_fn2',loss.item(),epoch)
	# /home/dipesh/PyTorch/CIFAR_100/cifar-100-python/pt_files
	# /home/dipesh/PyTorch/CIFAR_100/cifar-100-python/models
	# /home/dipesh/PyTorch/CIFAR_100/cifar-100-python/pickel_files
    timestamp = str(datetime.now())
    timestamp = timestamp[0:19]
    if (args.save_model):
        torch.save(model.state_dict(),"/home/dipesh/lab_work/pt_files/file:"+timestamp+" train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" CIFAR_100_cnn.pt")

    save_name_pkl = "/home/dipesh/lab_work/pickel_files/file:"+timestamp+"  train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.pkl"
    save_name_txt = "/home/dipesh/lab_work/models/file:"+timestamp+"  train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+" end.txt"
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

