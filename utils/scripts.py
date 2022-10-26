from math import log10, sqrt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

LABELS_TO_IDX = {"Breaked": 2, "Watermarked": 1, "Original": 0}
IDX_TO_LABELS = {0: "Original", 1: "Watermarked", 2: "Breaked"}

def PSNR(original, compressed):
    """
    Counts the peak signal-to-noise ration (PSNR) between the given images original and compressed
    """

    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def BER(a, b):
    """
    Counts the bit error rate (BER) between the given sequences a and b
    """

    ber = 0
    for x, y in zip(a, b):
        if x != y:
            ber += 1
    
    return ber / len(a)


def show(dataset, N=5, labels=None, figsize=(20, 20)):
    """ 
    Shows random N samples from the dataset
    """
    
    idxs = np.random.randint(0, len(dataset)-1, N)

    _, axs = plt.subplots(ncols=len(idxs), squeeze=False, figsize=figsize)

    for i, idx in enumerate(idxs):
        sample = dataset[idx]
        
        if isinstance(sample, tuple): # then it is in the form (x, y)
            sample, label = sample
            if isinstance(label, torch.TensorType):
                label = int(label.item())
            if labels:
                label = labels[label]
            axs[0, i].title.set_text(label)

        axs[0, i].imshow(F.to_pil_image(sample))
        axs[0, i].set(xticklabels = [], yticklabels = [], xticks = [], yticks = [])

    plt.show()

def evaluate(dataloader, model, device='cpu'):
    preds = []
    labels = []
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        for x_batch, y_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.tolist()
            
            outs = model(x_batch).detach().cpu()
            predictions = torch.argmax(torch.softmax(outs, 1), 1).tolist()
            
            # extend the `preds` and `labels` lists with predictions and true labels
            preds.extend(predictions)
            labels.extend(y_batch)
            
    report = classification_report(labels, preds, digits = 3)
    report_dict = classification_report(labels, preds, output_dict=True)
    print(report)
    return report_dict