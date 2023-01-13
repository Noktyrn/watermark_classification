import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
import os
from utils.scripts import PSNR, BER
import random
from bitstring import BitArray

from argparse import ArgumentParser
import yaml

WM_LENGTHS = [4, 5, 6, 7, 8]

parser = ArgumentParser()
parser.add_argument("-c", "--file", dest="filename",
                    help="YAML-config for the dataset creation", metavar="FILE")

args = parser.parse_args()

with open(args.filename, "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)

random.seed(data['seed'])

TRAIN_DIR = data['dataset']['train_dir']
VAL_DIR = data['dataset']['val_dir']

METHODS = data['dataset']['methods']

PSNR_BORDER = data['dataset'].get('psnr_border', 40)
BER_BORDER = data['dataset'].get('ber_border', 0.2)

DATASET_FOLDER = data['dataset']['target_dir']

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)
    os.mkdir(os.path.join(DATASET_FOLDER, '0'))
    os.mkdir(os.path.join(DATASET_FOLDER, '1'))
    os.mkdir(os.path.join(DATASET_FOLDER, '2'))

with open(DATASET_FOLDER + '/' + "conf.yaml", 'w') as yamlfile:
    res = yaml.dump(data, yamlfile)

DIM = (256, 256)

count_good = 0
count_bad = 0

for save_mode, dir in zip(['val', 'train'], [VAL_DIR, TRAIN_DIR]):
    for root, dirs, files in os.walk(dir):
        print(root)
        folder_name = root.split('/')[-1]

        for file in files:
            if file[-5:] == '.JPEG':
                ENC_LEN = random.choice(WM_LENGTHS)
                DEC_LEN = random.choice(WM_LENGTHS)

                encoder = WatermarkEncoder()
                original_decoder = WatermarkDecoder('bytes', ENC_LEN * 8)
                breaker = WatermarkEncoder()
                breaker_decoder = WatermarkDecoder('bytes', DEC_LEN * 8)

                ORIGINAL_WATERMARK = ''.join(random.choice('01') for _ in range(ENC_LEN * 8))
                encoder.set_watermark('bytes', BitArray(bin=ORIGINAL_WATERMARK).tobytes())

                BREAKING_WATERMARK = ''.join(random.choice('01') for _ in range(DEC_LEN * 8))
                breaker.set_watermark('bytes', BitArray(bin=BREAKING_WATERMARK).tobytes())

                bgr = cv2.imread(os.path.join(root, file))
                bgr = cv2.resize(bgr, DIM, interpolation = cv2.INTER_AREA)
                method = random.choice(METHODS)

                bgr_encoded = encoder.encode(bgr, method)
                watermark = original_decoder.decode(bgr_encoded, method)

                bgr_breaked = breaker.encode(bgr_encoded, method)
                watermark_b = breaker_decoder.decode(bgr_breaked, method)

                wm_ber = BER(ORIGINAL_WATERMARK, BitArray(watermark).bin)
                breaking_wm_ber = BER(BREAKING_WATERMARK, BitArray(watermark_b).bin)

                if PSNR(bgr, bgr_encoded) > PSNR_BORDER and wm_ber < BER_BORDER and breaking_wm_ber < BER_BORDER:
                    count_good += 1

                    cv2.imwrite(os.path.join(DATASET_FOLDER, '0', '0_'+file), bgr)
                    cv2.imwrite(os.path.join(DATASET_FOLDER, '1', '1_'+file), bgr_encoded)
                    cv2.imwrite(os.path.join(DATASET_FOLDER, '2', '2_'+file), bgr_breaked)
                else:
                    count_bad += 1

print("The number of images with PSNR > {} and BER > {}: {} \nThe number of other images {}".format(PSNR_BORDER, BER_BORDER,count_good, count_bad))