from argparse import ArgumentParser
import random
import os

import yaml
from torch import nn
import torch
import matplotlib.pyplot as plt

from utils.training import get_dataloaders, train, adjust_lr
from utils.model_builder import Model
from utils.scripts import evaluate

parser = ArgumentParser()
parser.add_argument("-c", "--file", dest="filename",
                    help="YAML-config for the training", metavar="FILE")
parser.add_argument("-t", "--folder", dest="foldername",
                    help="Name of the folder for the results", metavar="FOLDER")

args = parser.parse_args()

with open(args.filename, "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)

train_proportion = data.get('train_proportion', 0.8)
batch_size = data.get('batch_size', 32)
shuffle = data.get('shuffle', True)

data_path = data['dataset_path']

with open(data_path + '/' + 'conf.yaml', "r") as yamlfile:
    dataset_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    random.seed(dataset_conf.get('seed', 42))

train_loader, val_loader = get_dataloaders(data_path, train_proportion, batch_size, shuffle)

if not os.path.exists(args.foldername):
    os.makedirs(args.foldername)

with open(args.foldername + '/data_conf.yaml', "w") as yamlfile:
    res = yaml.dump(dataset_conf, yamlfile)

with open(args.foldername + '/train_conf.yaml', "w") as yamlfile:
    res = yaml.dump(data, yamlfile)

model_config = {'name':data['model_name'], 'head_only': data['head_only']}
net = Model(model_config)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer_name = data['optimizer']['name']
optimizer_params = data['optimizer'].get('params', dict())

base = data.get('lr_base', 1e-4)
step = data.get('lr_step', 0.999)
n_epochs = data.get('n_epochs', 4)

optimizer_params['lr'] = base
print(optimizer_params)

training_loss, validation_loss, model = train(net, optimizer_name, optimizer_params, loss_fn, train_loader, val_loader, 
                                              args.foldername+'/models', lr_adjuster=adjust_lr, return_model=True, device=device, 
                                              epochs=n_epochs, lr_base=base, lr_step=step)

plt.title('Training performance')
plt.plot(training_loss, label='training loss')
plt.plot(validation_loss, label='validation loss')
plt.legend()
plt.savefig(args.foldername + '/training.png')

test_report = evaluate(val_loader, model, device=device)

with open(args.foldername + '/results.yaml', "w") as yamlfile:
    res = yaml.dump(test_report, yamlfile)
