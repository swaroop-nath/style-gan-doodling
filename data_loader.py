import torch
from torch.utils import data
from pathlib import Path
import numpy as np
import json
import cairocffi as cairo

class Dataset_JSON(data.Dataset):

    def __init__(self, folder, image_size, large_aug=False):
        super().__init__()
        min_sample_num = 10000
        self.folder = folder
        self.image_size = image_size
        self.large_aug = large_aug
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.json')]
        while len(self.paths) < min_sample_num:
            self.paths.extend(self.paths)
        # notice the real influence of the trans / scale is side / 512 (original side) because of scalling in rendering
        if not large_aug:
            self.rotate = [-1/12*np.pi, 1/12*np.pi]
            self.trans = 0.01
            self.scale = [0.9, 1.1]
        else:
            self.rotate = [-1/4*np.pi, 1/4*np.pi]
            self.trans = 0.05
            self.scale = [0.75, 1.25]
            self.line_diameter_scale = [0.25, 1.25]
        if 'bird' in folder:
            self.id_to_part = {0:'initial', 1:'eye', 4:'head', 3:'body', 2:'beak', 5:'legs', 8:'wings', 6:'mouth', 7:'tail'}
        elif 'generic' in folder or 'fin' in folder or 'horn' in folder:
            self.id_to_part = { 0:'initial',  1:'eye',  2:'arms',  3:'beak',  4:'mouth',  5:'body',  6:'ears',  7:'feet',  8:'fin', 
                         9:'hair',  10:'hands',  11:'head',  12:'horns',  13:'legs',  14:'nose',  15:'paws',  16:'tail', 17:'wings'}
        self.n_part = len(self.id_to_part)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        json_data = json.load(open(path))
        input_parts_json = json_data['input_parts']
        target_part_json = json_data['target_part']
        # sample random affine parameters
        theta = np.random.uniform(*self.rotate)
        trans_pixel = 512*self.trans
        translate_x = np.random.uniform(-trans_pixel, trans_pixel)
        translate_y = np.random.uniform(-trans_pixel, trans_pixel)
        scale = np.random.uniform(*self.scale)
        if self.large_aug:
            line_diameter = np.random.uniform(*self.line_diameter_scale)*16
        else:
            line_diameter = 16
        # apply random affine transformation
        affine_target_part_json= self.affine_trans(target_part_json, theta, translate_x, translate_y, scale)
        processed_img_partial = []
        affine_vector_input_part = []
        for i in range(self.n_part):
            key = self.id_to_part[i]
            affine_input_part_json = self.affine_trans(input_parts_json[key], theta, translate_x, translate_y, scale)
            affine_vector_input_part += affine_input_part_json
            processed_img_partial.append(self.processed_part_to_raster(affine_input_part_json, side=self.image_size, line_diameter=line_diameter))
        processed_img_partial.append(self.processed_part_to_raster(affine_vector_input_part, side=self.image_size, line_diameter=line_diameter))
        processed_img_partonly = self.processed_part_to_raster(affine_target_part_json, side=self.image_size, line_diameter=line_diameter)
        processed_img = self.processed_part_to_raster(affine_vector_input_part+affine_target_part_json, side=self.image_size, line_diameter=line_diameter)
        # RandomHorizontalFlip
        if np.random.random() > 0.5:
            processed_img = processed_img.flip(-1)
            processed_img_partial = torch.cat(processed_img_partial, 0).flip(-1)
            processed_img_partonly = processed_img_partonly.flip(-1)
        else:
            processed_img_partial = torch.cat(processed_img_partial, 0)
        return processed_img, processed_img_partial, processed_img_partonly

    def sample_partial_test(self, n):
        sample_ids = [np.random.randint(self.__len__()) for _ in range(n)]
        sample_jsons = [json.load(open(self.paths[sample_id]))for sample_id in sample_ids]
        samples = []
        samples_partial = []
        samples_partonly = []
        for sample_json in sample_jsons:
            input_parts_json = sample_json['input_parts']
            target_part_json = sample_json['target_part']
            img_partial_test = []
            vector_input_part = []
            for i in range(self.n_part):
                key = self.id_to_part[i]
                vector_input_part += input_parts_json[key]
                img_partial_test.append(self.processed_part_to_raster(input_parts_json[key], side=self.image_size))
            img_partial_test.append(self.processed_part_to_raster(vector_input_part, side=self.image_size))
            samples_partial.append(torch.cat(img_partial_test, 0))
            img_partonly_test = self.processed_part_to_raster(target_part_json, side=self.image_size)
            img_test = self.processed_part_to_raster(vector_input_part+target_part_json, side=self.image_size)
            samples.append(img_test)
            samples_partonly.append(img_partonly_test)
        return torch.stack(samples), torch.stack(samples_partial), torch.stack(samples_partonly)

    def affine_trans(self, data, theta, translate_x, translate_y, scale):
        rotate_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        affine_data = []
        for item in data:
            if len(item) == 0:
                continue
            affine_item = np.array(item) - 256.
            affine_item = np.transpose(np.matmul(rotate_mat, np.transpose(affine_item)))
            affine_item[:, 0] += translate_x
            affine_item[:, 1] += translate_y
            affine_item *= scale
            affine_data.append(affine_item + 256.)
        return affine_data

    def processed_part_to_raster(self, vector_part, side=64, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
        """
        render raster image based on the processed part
        """
        original_side = 512.
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(line_diameter)
        # scale to match the new size
        # add padding at the edges for the line_diameter
        # and add additional padding to account for antialiasing
        total_padding = padding * 2. + line_diameter
        new_scale = float(side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)
        raster_images = []
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for stroke in vector_part:
            if len(stroke) == 0:
                continue
            ctx.move_to(stroke[0][0], stroke[0][1])
            for x, y in stroke:
                ctx.line_to(x, y)
            ctx.stroke()
        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)
        return torch.FloatTensor(raster_image/255.)[None, :, :]