# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:17:19 2023

@author: Yishu Malhotra
"""

from paddleocr import PaddleOCR

class paddleOCR:
    def __init__(self):
        """
        Paddleocr used to extrcat text from various sections of the form. 
        It supports sveral languages.
        """ 
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
    def process(self, img_path):
        ocr_output = self.ocr_model.ocr(img_path, cls=True)
        flatten_data = [item for sublist in ocr_output for item in sublist]
        converted_data = [{'bbox': item[0], 'text': item[1][0], 'confidence': item[1][1]} 
                          for item in flatten_data]
        return converted_data
