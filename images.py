import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

# Load a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    return torch.load(filepath)

def id_to_class(class_to_idx):
    dic = {}
    for key, value in class_to_idx:
        dic[value] = key
        return dic
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    im = Image.open(image)
    width, height = im.size
    
    if width > height:
        im.thumbnail((10000,256))
    else:
        im.thumbnail((256,10000))
    
    # Crop out the center 224x224 portion of the image.
    left = (im.size[0] - 224)/2
    top = (im.size[1] - 224)/2
    right = (im.size[0] + 224)/2
    bottom = (im.size[1] + 224)/2
    im = im.crop((left, top, right, bottom))
    
    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
    im = np.array(im)/255
    
    # As before, the network expects the images to be normalized in a specific way. 
    #For the means, it's [0.485, 0.456, 0.406] 
    means = [0.485, 0.456, 0.406]
    #and for the standard deviations [0.229, 0.224, 0.225]. 
    std_devs = [0.229, 0.224, 0.225]
    #You'll want to subtract the means from each color channel, then divide by the standard deviation.
    im = (im - means)/std_devs
    # And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. 
    # You can reorder dimensions using [`ndarray.transpose`]. 
    # The color channel needs to be first and retain the order of the other two dimensions    
    
    return im.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, idx_to_class, topk, cat_to_name_file, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    model.to(device)
    image = process_image(image_path)
    image = torch.FloatTensor(image)
    image = image.unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():

        #image = image.unsqueeze_(0)
        # TODO: Implement the code to predict the class from an image file
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk, dim=1)
        
        labels = []
        for item in classes[0].tolist():
            labels.append(idx_to_class[item])
        
        if cat_to_name_file:
            with open(cat_to_name_file, 'r') as f:
                cat_to_name = json.load(f)
            return [cat_to_name[str(i)] for i in labels]
        
        else:
            return labels
