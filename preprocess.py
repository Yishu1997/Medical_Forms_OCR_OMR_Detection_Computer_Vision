# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:14:45 2023

@author: Yishu Malhotra
"""
import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

class Preprocess:
  '''def __init__(self, img_paths: list):
    self.img_paths = img_paths'''

  def pdf_to_image(self, img_paths):
    """
    Function that converts pdfs to images and returns the a list of images
    as numpy arrays that can be used with opencv
    """
    images = []
    # convert PDF to image then to array ready for use with opencv
    for i in range(len(img_paths)):
      pages = convert_from_path(img_paths[i])
      images.append(np.array(pages[0]))
    return images

  def narrow(self, image, kernel, N, convert_color = False, binarize = True):
    """
    Draws narrower bounding boxes by heavily dilating the image 
    and picking out the first N largest blocks
    """
    #print(convert_color)
    #print('binarize: ', binarize)
    original = image.copy()
    if convert_color:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if binarize:
      #print(binarize)
      _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      #plt.imshow(image, cmap='gray')
      if np.mean(image) > 127:
        binary = cv2.bitwise_not(image)


    box_kernel = np.ones(kernel, np.uint8)
    dilation = cv2.dilate(image, box_kernel, iterations = 1)
    plt.imshow(dilation, cmap='gray')
    bounds, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL #cv2.RETR_TREE
    #print("hierarchy: ", hierarchy)

    boxes = []
    narrow_countours_info = []

    for cnt in bounds:
        x, y, w, h = cv2.boundingRect(cnt)
        '''if binarize:
          original = image'''
        region = original[y:y + h, x:x + w]
        boxes.append(region)
        narrow_countours_info.append([x, y, w, h])

    boxes = sorted(boxes, key=lambda i: -1 * i.shape[0] * i.shape[1])
    narrow_countours_info = sorted(narrow_countours_info, key=lambda i: -1 * i[2] * i[3])
    areas = [i.shape[0] * i.shape[1] for i in boxes]
    return boxes[:N],  narrow_countours_info[:N], areas[:N] #[:3]

  def narrow_images(self, images: list, kernel: tuple = (5, 25), N: int = 12, convert_color=False, binarize=False):
    """Function that takes as an input list of images and returns the following:
       narrowed_n: list of lists consisting of bboxes(cropped image sections) for each image in the list images.
       narrow_countours_info_n: list of lists consisting of countour information for each image in the list images.
       areas_n: list of lists consisting of areas of bboxes(cropped image sections) for each image in the list images.
    """
    narrowed_n = []
    narrow_countours_info_n = []
    areas_n = []
    for img in images:
        regions, narrow_countours_info, areas = self.narrow(img, kernel, N, convert_color, binarize)
        #for region in regions:
        narrowed_n.append(regions)
        narrow_countours_info_n.append(narrow_countours_info)
        areas_n.append(areas)
    return narrowed_n, narrow_countours_info_n, areas_n

  def binarize_images(self, images: list, black_on_white=False):
    """
    Function that takes a list of images and returns a list of binarized images
    """
    binarized = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            plt.imshow(binary)

        binarized.append(binary)
    return binarized

  def detect_template(self, areas: list, area_threshold: int =1800000):
    """
    A function that takes a list of lists of areas belonging to bboxes(cropped image sections)
    for each image and returns a list of templates to which the images belong.
    """
    template = []
    for i in range(len(areas)):
      if areas[i][2] < area_threshold:
        template.append('template_1')
      else:
        template.append('template_2')
    return template