# Imports here
import torch
import argparse
import run_model

# Get data from command line
parser = argparse.ArgumentParser()
parser.add_argument('path', action="store", type=str, help = 'Name of directory where pictures are located, example = flowers')
parser.add_argument('--epochs', type=int, default = 2, help = 'Set epochs for nn model [0-3], default 1')
parser.add_argument('--learning_rate', type=float, default = 0.003, help = 'Set learning rate, default 0.003')
parser.add_argument('--arch', type=str, default = 'densenet121', help = 'Set the torchvision model used for prediction, default = densenet121')
parser.add_argument('--hidden_units', type=int, default = 256, help = 'set hidden units')

path = parser.parse_args().path.strip('/')
epochs = parser.parse_args().epochs
learning_rate = parser.parse_args().learning_rate
arch = parser.parse_args().arch.strip('"')
hidden_units = parser.parse_args().hidden_units

#load data
trainloader, validloader, testloader, class_to_idx = run_model.load_data(path)
# train model
run_model.train(trainloader, validloader, class_to_idx, epochs, learning_rate, arch, hidden_units)
# test model
run_model.test(testloader)
print('Done training')

#and we are done