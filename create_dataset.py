import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
from difflib import SequenceMatcher
import os
from utils.scripts import PSNR
import random

from argparse import ArgumentParser
import yaml

parser = ArgumentParser()
parser.add_argument("-c", "--file", dest="filename",
                    help="YAML-config for the dataset creation", metavar="FILE")

args = parser.parse_args()

with open(args.filename, "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)

random.seed(data['seed'])

TRAIN_DIR = data['dataset']['train_dir']
VAL_DIR = data['dataset']['val_dir']
ORIGINAL_WATERMARK = data['dataset']['original_wm']
BREAKING_WATERMARK = data['dataset']['breaking_wm']
PSNR_BORDER = data['dataset']['psnr_border']

DATASET_FOLDER = data['dataset']['target_dir']

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)
    os.mkdir(os.path.join(DATASET_FOLDER, '0'))
    os.mkdir(os.path.join(DATASET_FOLDER, '1'))
    os.mkdir(os.path.join(DATASET_FOLDER, '2'))

with open(DATASET_FOLDER + '/' + "conf.yaml", 'w') as yamlfile:
    res = yaml.dump(data, yamlfile)

DIM = (256, 256)

encoder = WatermarkEncoder()
encoder.set_watermark('bytes', ORIGINAL_WATERMARK.encode("UTF-8"))
original_decoder = WatermarkDecoder('bytes', len(ORIGINAL_WATERMARK)*8)

breaker = WatermarkEncoder()
breaker.set_watermark('bytes', BREAKING_WATERMARK.encode("UTF-8"))
breaker_decoder = WatermarkDecoder('bytes', len(BREAKING_WATERMARK)*8)

count_good = 0
count_bad = 0

for save_mode, dir in zip(['val', 'train'], [VAL_DIR, TRAIN_DIR]):
    for root, dirs, files in os.walk(dir):
        print(root)
        folder_name = root.split('/')[-1]

        for file in files:
            if file[-5:] == '.JPEG':
                bgr = cv2.imread(os.path.join(root, file))
                bgr = cv2.resize(bgr, DIM, interpolation = cv2.INTER_AREA)

                bgr_encoded = encoder.encode(bgr, 'dwtDct')
                watermark = original_decoder.decode(bgr_encoded, 'dwtDct')

                bgr_breaked = breaker.encode(bgr_encoded, 'dwtDct')
                watermark_b = breaker_decoder.decode(bgr_breaked, 'dwtDct')

                wm_ratio = SequenceMatcher(None, watermark, ORIGINAL_WATERMARK.encode("UTF-8")).ratio()
                breaking_wm_ratio = SequenceMatcher(None, watermark_b, BREAKING_WATERMARK.encode("UTF-8")).ratio()

                if PSNR(bgr, bgr_encoded) > PSNR_BORDER:
                    count_good += 1

                    cv2.imwrite(os.path.join(DATASET_FOLDER, '0', '0_'+file), bgr)
                    cv2.imwrite(os.path.join(DATASET_FOLDER, '1', '1_'+file), bgr_encoded)
                    cv2.imwrite(os.path.join(DATASET_FOLDER, '2', '2_'+file), bgr_breaked)
                else:
                    count_bad += 1

print("The number of images with PSNR > {}: {} \nThe number of other images {}".format(PSNR_BORDER, count_good, count_bad))