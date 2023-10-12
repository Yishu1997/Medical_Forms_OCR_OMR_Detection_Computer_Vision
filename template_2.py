# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:22:18 2023

@author: Yishu Malhotra
"""

import cv2
import numpy as np
from difflib import SequenceMatcher
from ocr import paddleOCR
from preprocess import Preprocess

class Template_2:
  def __init__(self, image):
    self.image = image
    self.prep = Preprocess()
    self.ocr = paddleOCR()
    self.narrowed, self.narrow_contours_info, self.areas = self.get_sub_narrowed_data()
    print('self.narrowed: ', len(self.narrowed))

  def get_sub_narrowed_data(self):
    binarized = self.prep.binarize_images([self.image], black_on_white=False)
    narrowed, narrow_contours_info, areas = self.prep.narrow_images(binarized, kernel=(5,5), N=15)
    return narrowed[0], narrow_contours_info[0], areas[0]

  def ocr_concat_data(self, ocr_output, indentation = ' ', discard_checkbox_data = None):
    concatenated_str = ''
    confidence = []
    for data_dict in ocr_output:
      if discard_checkbox_data is None:
        concatenated_str = concatenated_str + data_dict['text'] + indentation
        confidence.append(data_dict['confidence'])
      elif data_dict['text'] != discard_checkbox_data:
        concatenated_str = concatenated_str + data_dict['text'] + indentation
        confidence.append(data_dict['confidence'])
      else:
        pass

    return [concatenated_str, np.mean(confidence)]

  def extract_physician_information(self):
    ocr_output = self.ocr.process(self.narrowed[5])
    physician_name = [ocr_output[0]['text'], ocr_output[0]['confidence']]
    date = [ocr_output[1]['text'], ocr_output[1]['confidence']]
    practioner_number = [ocr_output[4]['text'], ocr_output[4]['confidence']]
    # Need form with data of Copy to for finding its position in ocr_output
    copy_to = ''
    physician_information = {
                             'physician_name': physician_name,
                             'date': date,
                             'practioner_number': practioner_number,
                             'copy_to': copy_to
                             }

    return physician_information

  def extract_xray(self):
    ocr_output = self.ocr.process(self.narrowed[6])
    if len(ocr_output) > 1:
      concatenated_str = self.ocr_concat_data(ocr_output[1:])
    else:
      concatenated_str = None
    return concatenated_str

  def extract_ultrasound(self):
    ocr_output = self.ocr.process(self.narrowed[7])
    if len(ocr_output) > 1:
      concatenated_str = self.ocr_concat_data(ocr_output[1:])
    else:
      concatenated_str = None
    return concatenated_str

  def extract_referring_physician_office_stamp(self):
    ocr_output = self.ocr.process(self.narrowed[8])
    concatenated_str = self.ocr_concat_data(ocr_output[1:])
    return concatenated_str

  def extract_hysterosalpingogram(self):
    # No data in this section in the forms we have yet, so solution based on 
    # what we see adn theoritical possibilities.
    ocr_output = self.ocr.process(self.narrowed[9])
    concatenated_str = self.ocr_concat_data(ocr_output[2:], indentation=" ")
    return concatenated_str

  def calc_similarity_confidence(self, label_a: str, label_list: list):
    similarity_confidence = []
    for i in range(len(label_list)):
      similarity_confidence.append(SequenceMatcher(None, label_a, label_list[i]).ratio())
    return np.array(similarity_confidence)

  def detect_patient_information_label_and_associated_data(self, label_data: str):
    labels = ["Patient's", 'LastName', "Patient's Last Name", "Patient's LastName", 'Last', 'Name',
          'Address', 'City', 'Cty', 'Gty', 'First', "First Name", "FirstName", 'Postal', 'Fostal',
          'PostalCode', 'Postal Code', 'Code', 'Phone', 'Number', "PhoneNumber", "Phone Number",
          'Sex:', 'Sex', ':', 'Date', 'of', 'Birth', "Date of Birth","Dateof Birth", "DateofBirth"]
    x = label_data['text'].split()
    associated_data = []
    for i in range(len(x)):
      if  x[i] not in labels:
        associated_data.append(x[i])
    label_data_text_copy = label_data['text']
    for j in range(len(associated_data)):
      extracted_label = label_data_text_copy.replace(associated_data[j],"")
      label_data_text_copy = extracted_label

    """similarity_confidence = []
    for i in range(len(labels)):
      similarity_confidence.append(SequenceMatcher(None, extracted_label, labels[i]).ratio())"""
    """if extracted_label is None:
      return [None, None, None]"""
    similarity_confidence = self.calc_similarity_confidence(extracted_label, labels)
    max_similarity_confidence = similarity_confidence.max()
    index_of_detected_label = similarity_confidence.argmax()
    detected_label =  labels[index_of_detected_label]
    if max_similarity_confidence < 0.5:
      detected_label = None

    if len(associated_data) > 1:
      final_associated_data = ''
      for j in range(len(associated_data)):
        final_associated_data = final_associated_data + associated_data[j] + " "
    elif len(associated_data) == 1:
      final_associated_data = associated_data[0]
    else:
      final_associated_data = None
    return [detected_label, final_associated_data, label_data['confidence']]

  def extract_patients_information(self):
    x,y,w,h = self.narrow_contours_info[2]
    x2,y2,w2,h2 = self.narrow_contours_info[10]
    gray = cv2.cvtColor(self.image[y2:y,:], cv2.COLOR_RGB2GRAY)
    ocr_output = self.ocr.process(gray)
    business_number_present = False
    health_card_idx = 17
    if len(ocr_output) > 19:
      business_number_present = True
    if business_number_present:
      business_number_idx = 12
      health_card_idx = health_card_idx + 1

    patient_info_dict = {
                        "Patient's Last Name": [ocr_output[0]['text'], ocr_output[0]['confidence']],
                        "First Name": [ocr_output[1]['text'], ocr_output[1]['confidence']],
                        "Sex(M/F/X)": [ocr_output[2]['text'], ocr_output[2]['confidence']],
                        "Date of Birth": [ocr_output[3]['text'], ocr_output[3]['confidence']],
                        "Address": [ocr_output[8]['text'], ocr_output[8]['confidence']],
                        "City": [ocr_output[9]['text'], ocr_output[9]['confidence']],
                        "Postal Code": [ocr_output[10]['text'], ocr_output[10]['confidence']],
                        "Home Number": [ocr_output[11]['text'], ocr_output[11]['confidence']],
                        "Business Number": "", #[ocr_output[12]['text'], ocr_output[12]['confidence']],
                        "Health Card Number": [ocr_output[health_card_idx]['text'], ocr_output[health_card_idx]['confidence']]
                        }

    return patient_info_dict

  def checkbox_detection(self, image):
    """
    This function takes as an input a grayscale image and
    detects checkboxes in an image andd returns a list
    of [x,y,w,h] for each detected checkbox
    """
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold

    # Can rather pass grayscale cropped image
    #img = image[y_i:y_i+h_i, x_i:x_i+w_i]
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #x_gray.copy()
    img = image.copy()
    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area filtering to remove noise
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    AREA_THRESHOLD = 100
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            cv2.drawContours(thresh, [c], -1, 0, -1)

    # Repair checkbox horizontal and vertical walls
    repair_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    repair = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, repair_kernel1, iterations=1)
    repair_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, repair_kernel2, iterations=1)

    # Detect checkboxes using shape approximation and aspect ratio filtering
    checkbox_contours = []
    cnts, _ = cv2.findContours(repair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    checkbox_cordinates = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2):
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3) #original
            checkbox_contours.append(c)
            checkbox_cordinates.append([x,y,w,h])

    #print('Checkboxes:', len(checkbox_contours))
    #print("checkbox_cordinates: ",checkbox_cordinates)
    return checkbox_cordinates

  def checkbox_mark_recognition(self, image):
    """
    This function takes as an input a grayscale image.
    It returns a list of cordinates of the
    marked or ticked checkboxes
    """
    # Detecting marked checkboxes and getting its coordinates
    checkbox_cordinates = self.checkbox_detection(image)
    areas_checkbox = []
    average_color = []
    weighted_avg_color = []
    marked_checkbox_cordinates = []
    for i in range(len(checkbox_cordinates)):
      x_c,y_c,w_c,h_c = checkbox_cordinates[i]
      checkbox_img_gray = image[y_c:y_c+h_c, x_c:x_c+w_c] #cv2.cvtColor(image[y_c:y_c+h_c, x_c:x_c+w_c], cv2.COLOR_BGR2GRAY)
      areas_checkbox.append(checkbox_img_gray.shape[0] * checkbox_img_gray.shape[1])
      average_color.append(checkbox_img_gray.mean(axis=0).mean(axis=0))

    normalized_areas_checkbox = [float(i)/max(areas_checkbox) for i in areas_checkbox]

    for i in range(len(average_color)):
      w_avg_color = average_color[i] * normalized_areas_checkbox[i]
      weighted_avg_color.append(w_avg_color)
      # Putting a threshold of 128 to detect marked checkboxes
      if w_avg_color <= 128:
        marked_checkbox_cordinates.append(checkbox_cordinates[i])

    return marked_checkbox_cordinates

  def vertical_checkboxes_associated_data_extraction(self, image):
    """
    This function takes a grayscale image and the coordinates of
    marked checkboxes as an input and returns associated data of marked checkboxes
    """
    # Vertical checkboxes associated date extraction
    # ASUMMING the main image sent to checkbox recognition is grayscale:
    marked_checkbox_cordinates = self.checkbox_mark_recognition(image)
    marked_checkbox_ocr_data = ''
    if len(marked_checkbox_cordinates) != 0:
      for i in range(len(marked_checkbox_cordinates)):
        x_c,y_c,w_c,h_c = marked_checkbox_cordinates[i]
        checkbox_text_area = image[y_c:y_c+h_c, x_c+w_c:]
        marked_checkbox_ocr_data = self.ocr.process(checkbox_text_area)
        #print("ocr_output: ", ocr_output)
        #marked_checkbox_ocr_data.append(ocr_output[0])
      return marked_checkbox_ocr_data
    else:
      return None

  def vertical_horizontal_checkboxes_associated_data_extraction(self, image):
    """
    This function takes a grayscale image and the coordinates of
    marked checkboxes as an input and returns associated data of marked checkboxes
    This functions takes care of cases where checboxes are aligned both vertically
    and horizontally.
    """
    # Vertical + Horizontal checkboxes associated date extraction
    # ASUMMING the main image sent to checkbox recognition is grayscale:
    marked_checkbox_cordinates = self.checkbox_mark_recognition(image)
    marked_checkbox_ocr_data = ''
    if len(marked_checkbox_cordinates) != 0:
      for i in range(len(marked_checkbox_cordinates)):
        x_c,y_c,w_c,h_c = marked_checkbox_cordinates[i]
        offset = int(image.shape[1]/2)
        checkbox_text_area = image[y_c:y_c+h_c, x_c+w_c:x_c+w_c+offset]
        ocr_output = self.ocr.process(checkbox_text_area)
        marked_checkbox_ocr_data = ocr_output
      return marked_checkbox_ocr_data
    else:
      return None

  def fluoroscopy(self):
    dict_fluoroscopy = {
                        "UGI": False,
                        "UGI with follow-through": False,
                        "Small Bowel": False
                        }
    fluoroscopy_labels = list(dict_fluoroscopy.keys())
    x_i,y_i,w_i,h_i = self.narrow_contours_info[13]
    gray_image_crop = cv2.cvtColor(self.image[y_i:y_i+h_i, x_i:x_i+w_i], cv2.COLOR_RGB2GRAY)
    marked_checkbox_ocr_data = self.vertical_checkboxes_associated_data_extraction(gray_image_crop)
    if marked_checkbox_ocr_data is None:
      return dict_fluoroscopy
    else:
      for i in range(len(marked_checkbox_ocr_data)):
        extracted_label = marked_checkbox_ocr_data[i]['text']
        similarity_confidence = self.calc_similarity_confidence(extracted_label, fluoroscopy_labels)
        index_of_detected_label = similarity_confidence.argmax()
        detected_label =  fluoroscopy_labels[index_of_detected_label]
        dict_fluoroscopy[detected_label] = True

      return dict_fluoroscopy

  def insurance(self):
    dict_insurance = {
                        "MSP": False,
                        "ICBC": False,
                        "Private": False,
                        "WorkSafe BC": False,
                        "Other:": False
                        }
    insurance_labels = list(dict_insurance.keys())
    x_i,y_i,w_i,h_i = self.narrow_contours_info[12]
    gray_image_crop = cv2.cvtColor(self.image[y_i:y_i+h_i, x_i:x_i+w_i], cv2.COLOR_RGB2GRAY)
    marked_checkbox_ocr_data = self.vertical_checkboxes_associated_data_extraction(gray_image_crop)
    if marked_checkbox_ocr_data is None:
      return dict_insurance
    else:
      if len(marked_checkbox_ocr_data) == 1:
        extracted_label = marked_checkbox_ocr_data[0]['text']
        similarity_confidence = self.calc_similarity_confidence(extracted_label, insurance_labels)
        index_of_detected_label = similarity_confidence.argmax()
        detected_label =  insurance_labels[index_of_detected_label]
        dict_insurance[detected_label] = True
      else:  # For the case Other is marked and has some text beside it
        #print(marked_checkbox_ocr_data)
        extracted_label = marked_checkbox_ocr_data[0]['text']
        similarity_confidence = self.calc_similarity_confidence(extracted_label, insurance_labels)
        index_of_detected_label = similarity_confidence.argmax()
        detected_label =  insurance_labels[index_of_detected_label]
        output = ''
        confidence = []
        for i in range(1, len(marked_checkbox_ocr_data)):
          output = output + " " + marked_checkbox_ocr_data[i]['text']
          confidence.append(marked_checkbox_ocr_data[i]['confidence'])
        dict_insurance[detected_label] = [output, np.mean(confidence)]

      return dict_insurance
  
  def pertinent_clinical_information(self):
    dict_pertinent = {
                        "Diabetic": False,
                        "Known/suspected communicable disease": False,
                        "Other (please specify):": False,
                        "Verbal": False
                        }
    insurance_labels = list(dict_pertinent.keys())
    x_i,y_i,w_i,h_i = self.narrow_contours_info[4]
    gray_image_crop = cv2.cvtColor(self.image[y_i:y_i+h_i, x_i:x_i+w_i], cv2.COLOR_RGB2GRAY)
    marked_checkbox_ocr_data = self.vertical_horizontal_checkboxes_associated_data_extraction(gray_image_crop)
    if marked_checkbox_ocr_data is None:
      return dict_pertinent
    else:
      if len(marked_checkbox_ocr_data) == 1:
        extracted_label = marked_checkbox_ocr_data[0]['text']
        similarity_confidence = self.calc_similarity_confidence(extracted_label, insurance_labels)
        index_of_detected_label = similarity_confidence.argmax()
        detected_label =  insurance_labels[index_of_detected_label]
        dict_pertinent[detected_label] = True
      else:  # For the case Other is marked and has some text beside it
        #print(marked_checkbox_ocr_data)
        extracted_label = marked_checkbox_ocr_data[0]['text']
        similarity_confidence = self.calc_similarity_confidence(extracted_label, insurance_labels)
        index_of_detected_label = similarity_confidence.argmax()
        detected_label =  insurance_labels[index_of_detected_label]
        output = ''
        confidence = []
        for i in range(1, len(marked_checkbox_ocr_data)):
          output = output + " " + marked_checkbox_ocr_data[i]['text']
          confidence.append(marked_checkbox_ocr_data[i]['confidence'])
        dict_pertinent[detected_label] = [output, np.mean(confidence)]

      return dict_pertinent