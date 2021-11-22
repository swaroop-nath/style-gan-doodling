import imagehash
from PIL import Image
import os
import numpy as np
from numpy.random import choice, seed
from tqdm import tqdm

seed(42)

def compute_gen_div(img_path_1, img_path_2, **kwargs):
    hash1 = imagehash.average_hash(Image.open(img_path_1), **kwargs)
    hash2 = imagehash.average_hash(Image.open(img_path_2), **kwargs)

    # References - 
    # 1. https://people.cs.umass.edu/~liberato/courses/2020-spring-compsci590k/lectures/09-perceptual-hashing/
    # 2. https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
    return hash1 - hash2 # Returns the hamming distance between the two

if __name__ == '__main__':
    indices = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 0)]
    sample = 1000

    # with open('gd_scores.log', 'w') as file:
    #     file.write('GD Scores for sample size {}'.format(sample))

    for idx_1, idx_2 in indices:
        img_base_path = 'gen_imgs/bird_{}.{}/DoodlerGAN_all/bw'.format(idx_1, idx_2)

        file_paths = os.listdir(img_base_path)
        chosen_files = choice(file_paths, sample, replace=False)

        gd_score = 0 # higher is better
        for i in tqdm(range(sample)):
            for j in range(sample):
                if i == j: continue
                img_path_1 = os.path.join(img_base_path, chosen_files[i])
                img_path_2 = os.path.join(img_base_path, chosen_files[j])
                hamming_distance = compute_gen_div(img_path_1, img_path_2)
                gd_score += hamming_distance

        gd_score /= sample

        with open('gd_scores.log', 'a') as file:
            file.write('GD Score for {}: {:0.2f}\n'.format(img_base_path.split('/')[1], gd_score))