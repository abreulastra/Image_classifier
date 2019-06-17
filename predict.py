#This file contains all the functions relevant to run predict.py
import images
import argparse

#image_dir = 'flowers/train/10/image_07086.jpg'

# Get data from command line
parser = argparse.ArgumentParser()
parser.add_argument('path', action="store", type=str, help = 'Name of directory where picture is located, example = flowers/train/10/image_07086.jpg')
parser.add_argument('model', action="store", type=str, help="Name of file containing model at checkpoint")
parser.add_argument('--top_k', type=int, default = 3, help = 'Get top_k likeliest classes [0-5], default 1')
parser.add_argument('--category_names', default=None, help="Provide file that maps categories to real names, example=cat_to_name.json")
parser.add_argument('--gpu', action="store_true", help="Uses GPU for inference, if available") 
image_dir = parser.parse_args().path.strip('/')
cp_file = parser.parse_args().model
top_k = parser.parse_args().top_k
cat_to_name_file = parser.parse_args().category_names
gpu = parser.parse_args().gpu
                    
#loading model
model_c = images.load_checkpoint(cp_file + '.pth')
idx_to_class = {}
for key, value in model_c.class_to_idx.items():
    idx_to_class[value] = key

#getting results
results = images.predict(image_dir, model_c, idx_to_class, top_k, cat_to_name_file, gpu)
print(results)