import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm
import os

OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'RMSProp': torch.optim.RMSprop,
    'SGD': torch.optim.SGD
}

def get_optimizer(net, name, optim_params):
    opt = OPTIMIZERS[name]
    return opt(filter(lambda p: p.requires_grad, net.parameters()), **optim_params)

def adjust_lr(optimizer, epoch, base_lr, lr_step):
    lr = base_lr
    for _ in range(epoch):
        lr *= lr_step
    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr

def train(model, optim_name, optim_params, loss_fn, 
          train_loader, val_loader, save_path, return_model=False,
          device='cpu', lr_adjuster=None, lr_base=1e-4, lr_step=0.999, epochs=10):
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    
    training_losses = []
    validation_losses = [] 
    
    # initialize the model
    optimizer = get_optimizer(model, optim_name, optim_params)

    # pass the model to the given device
    model.to(device)

    print("Number of samples")
    print("Training:", n_train)
    print('Validation:', n_val)
    for epoch in range(epochs):
        # define running losses
        epoch_training_running_loss = 0
        epoch_val_running_loss = 0
        
        bar = tqdm(enumerate(train_loader), total=n_train//train_loader.batch_size)

        # loop through every batch in the training loader
        for batch_idx, (x_batch, y_batch) in bar:
            if lr_adjuster:
                step = epoch * len(bar) + batch_idx
                lr = lr_adjuster(optimizer, step, lr_base, lr_step)

            # pass the batches to given device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # set the gradients to 0 beforehand
            # it can also be written after `optimizer.step()`, just a preference.
            outs = model(x_batch)
            loss = loss_fn(outs, y_batch)
            # calculate the gradients and apply an optimization step
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

            # we can use `.item()` method to read the loss value
            # since the loss function automatically calculates the loss by averaging the input size,
            # we will multiply it with the batch size to add it
            # then we can average it by the whole dataset size
            # note: it is also possible to average the loss by the number of batches at the end of the epoch (without multiplying with x_batch.size(0))
            # but this approach is more straightforward.
            epoch_training_running_loss += (loss.item() * x_batch.size(0))

        with torch.no_grad():
            model.eval()
            vbar = tqdm(val_loader, total=n_val//val_loader.batch_size)
            for x_batch, y_batch in vbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                outs = model(x_batch)
                loss = loss_fn(outs, y_batch)
                    
                epoch_val_running_loss += (loss.item() * x_batch.size(0))
            model.train()

        average_training_loss = epoch_training_running_loss / n_train
        average_validation_loss = epoch_val_running_loss / n_val

        training_losses.append(average_training_loss)
        validation_losses.append(average_validation_loss)
        
        if lr_adjuster:
            print(f"epoch {epoch+1}/{epochs}, lr={lr} | avg. training loss: {average_training_loss:.3f}, avg. validation loss: {average_validation_loss:.3f}")
        else:
            print(f"epoch {epoch+1}/{epochs} | avg. training loss: {average_training_loss:.3f}, avg. validation loss: {average_validation_loss:.3f}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'/epoch_{}_result.pth'.format(epoch))
            
    # return the training and validtion losses, also return the model if return_model is True
    if return_model:
        return training_losses, validation_losses, model
    else:
        return training_losses, validation_losses

def get_dataloaders(data_path, proportion=0.8, batch_size=16, shuffle=True):
    """
    Returns two dataloaders (for training and validation), created from the ImageFolder dataset, located by the data_path address.
    Splits these dataloaders using the given proportion.
    """

    transformations = transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224),  
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = ImageFolder(data_path, transform=transformations)

    num_train = int(len(dataset) * proportion)
    num_val = len(dataset) - num_train

    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train_set, batch_size, shuffle)
    val_dataloader = DataLoader(val_set, batch_size, shuffle)
    return train_dataloader, val_dataloader