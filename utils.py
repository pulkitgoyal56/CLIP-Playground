#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from itertools import product

import numpy as np

from PIL import Image, ImageDraw, ImageOps
import skimage

import matplotlib.pyplot as plt

import logging

import constants


class Im():
    BASE_DIR = Path('data/sketch')

    def __init__(self, basename, category, prefix=''):
        self.category = category
        self.dirname = Im.BASE_DIR / self.category / prefix
        self.basename = basename
        self.path = self.dirname / self.basename
        if self.path.is_file() and self.path.suffix=='.png':
            self.image = Image.open(self.path)

    def apply(self, func, *args, **kwargs):
        self.image = func(self.image, *args, **kwargs)
        return self

    def save(self, prefix=''):
        filepath = self.dirname / prefix / self.basename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(filepath)

    @staticmethod
    def pixelate(image, pixel_size=constants.GRIDCELL_SIZE):
        return image.resize(tuple(map(lambda s: int(s / pixel_size), image.size)), resample=Image.Resampling.BILINEAR).convert('1').resize(image.size, Image.Resampling.NEAREST)

    def preprocess(self, invert=True, step=constants.GRIDCELL_SIZE, *, save=False):
        if invert:
            self.apply(ImageOps.invert)
        my_grid = ShapeGridworld(gridcell_size=step)
        my_grid.register(self.image)
        self.image = my_grid.render()
        if save:
            self.save(f'pre-processed_{step}' if step else '')

    def show(self, figsize=(10, 10)):
        fig = plt.figure(figsize=figsize)
        plt.axis("off")
        plt.imshow(np.array(self.image), vmin=0, vmax=1)
        plt.show()

    def _ipython_display_(self):
        if self.image:
            display(self.image)

class ShapeGridworld():
    COLOR = 1 # (0.21568627450980393, 0.49411764705882355, 0.7215686274509804)

    def __init__(self, width=constants.GRID_WIDTH, height=constants.GRID_HEIGHT, gridcell_size=constants.GRIDCELL_SIZE, shape='circle'):
        self.width = width
        self.height = height
        self.shape = shape
        self.gridcell_size = gridcell_size
        self.objects = np.zeros((self.height, self.width))
        self.create_kernel()

    def create_kernel(self):
        if self.shape == 'square':
            self.kernel = np.ones((self.gridcell_size, self.gridcell_size))
        elif self.shape == 'circle':
            radius = int(self.gridcell_size/2)
            center = (radius,) * 2

            y, x = np.ogrid[:self.gridcell_size, :self.gridcell_size]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

            self.kernel = dist_from_center <= radius

    def register(self, image):
        image = np.array(image.resize((self.width*self.gridcell_size, self.height*self.gridcell_size)))
        for i, j in product(range(self.height), range(self.width)):
            self.objects[i, j] = bool(np.tensordot(image[self.gridcell_size*i:self.gridcell_size*(i + 1), self.gridcell_size*j:self.gridcell_size*(j + 1)], self.kernel))

    def render(self):
        image = np.zeros((self.height*self.gridcell_size, self.width*self.gridcell_size), dtype=bool) # , 3), dtype=np.float32)
        for pos, val in np.ndenumerate(self.objects):
            if val:
                if self.shape == 'circle':
                    # image = Image.fromarray(image)
                    # ImageDraw.Draw(image).ellipse((pos[0]*self.gridcell_size, pos[1]*self.gridcell_size,  (pos[0] + 1)*self.gridcell_size, (pos[0] + 1)*self.gridcell_size), fill=ShapeGridworld.COLOR, outline=ShapeGridworld.COLOR)
                    # image = np.array(image)
                    rr, cc = skimage.draw.disk((pos[0]*self.gridcell_size + self.gridcell_size/2, pos[1]*self.gridcell_size + self.gridcell_size/2), self.gridcell_size/2, shape=image.shape)
                    image[rr, cc] = ShapeGridworld.COLOR
                elif self.shape == 'square':
                    image[self.gridcell_size*pos[0]:self.gridcell_size*(pos[0] + 1), self.gridcell_size*pos[1]:self.gridcell_size*(pos[1] + 1)] = ShapeGridworld.COLOR

        image[:, ::self.gridcell_size] = image[::self.gridcell_size, :] = 1 # [1, 1, 1]

        return Image.fromarray(image)

def create_mask(size, pixel_size=constants.GRIDCELL_SIZE):
    image = Image.new('1', size, 'black')
    width, height = size
    image_draw = ImageDraw.Draw(image)
    for x in range(0, width - pixel_size + 1, pixel_size):
        for y in range(0, height - pixel_size + 1, pixel_size):
            image_draw.ellipse((x, y, x + pixel_size, y + pixel_size), fill = 'white', outline ='white')
    return image

def gridcell_value(image, x, y, gridcell_size=constants.GRIDCELL_SIZE, shape='square'):
    image = np.array(image)
    if shape == 'square':
        return image[x-int(gridcell_size/2):x+int(gridcell_size/2), y-int(gridcell_size/2):y+int(gridcell_size/2)].any()
    elif shape == 'circle':
        for (i, j), v in np.ndenumerate(image[x-int(gridcell_size/2):x+int(gridcell_size/2), y-int(gridcell_size/2):y+int(gridcell_size/2)]):
            if (np.sqrt((i - gridcell_size/2)**2 + (j - gridcell_size/2)**2) < (gridcell_size/2)**2) and v:
                return True

def mask_grid(image, gridcell_size=constants.GRIDCELL_SIZE, method='mask'): # 'grid'
    if method == 'mask':
        return Image.fromarray(np.array(create_mask(image.size, gridcell_size)) * np.array(image))
    elif method == 'grid':
        grid_image = np.zeros(image.size, dtype=bool)
        for x, y in zip(*map(np.ravel, np.mgrid[0:image.size[0]:int(gridcell_size/2), 0:image.size[1]:int(gridcell_size/2)])):
            grid_image[x-int(gridcell_size/2):x+int(gridcell_size/2), y-int(gridcell_size/2):y+int(gridcell_size/2)] = gridcell_value(image, x, y, gridcell_size)
        return Image.fromarray(grid_image)

def shift_and_scale(image):
    pass

def show_images(filenames, category, results=None, gridcell_size=constants.GRIDCELL_SIZE):
    if not len(filenames):
        logging.warn('> No files to show.')
        return
    fig, axs = plt.subplots(int(np.ceil(len(filenames)/10)), 10, figsize=(20, int(np.ceil(len(filenames)/10))*2), sharex=True, sharey=True)
    fig.set_facecolor('black')
    axs = iter(axs.ravel())
    for filename in filenames[::-1]:
        ax = next(axs)
        ax.imshow(Im(f'{filename}.png', category, f'pre-processed_{gridcell_size}' if gridcell_size else None).image, cmap='binary')
        if results:
            ax.set_title(fr'$\bf{{{results[filename][0]:.2f}}}$ | {-results[filename] @ np.log(results[filename]):.2f}', color='white')
        ax.axis('off')
    for ax in axs:
        ax.axis('off')
