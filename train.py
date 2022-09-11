import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt

from run_utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg19', choices=['vgg19', 'vgg13'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training is starting')
def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        model.to(device)
        for inputs, labels in dataloaders['trainloader']:
            steps += 1
            # Move input and label tensors to the default device
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs2, label in dataloaders['validloader']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"valid loss: {valid_loss/len(dataloaders['validloader']):.3f}.. "
                      f"valid accuracy: {accuracy/len(dataloaders['validloader']):.3f}")
                running_loss = 0
def main():
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    
    'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]), 
    'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train_data' : datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid_data' : datasets.ImageFolder(valid_dir, data_transforms['valid']),
        'test_data' : datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64, shuffle=True),
        'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True)
    }
    
    model = getattr(models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    
    if args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
    
    
        model.classifier = nn.Sequential(nn.Linear(25088, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    elif args.arch == "vgg13": 
        model.classifier = nn.Sequential(nn.Linear(25088, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
        
        
        
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets['train_data'].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir  
    save_checkpoint(path, model, optimizer, args, model.classifier)


if __name__ == "__main__":
    main()
