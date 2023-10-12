# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:18:57 2023

@author: Yishu Malhotra
"""
import cv2
import numpy as np
from difflib import SequenceMatcher
from ocr import paddleOCR
from preprocess import Preprocess

class Template_1:
  def __init__(self, image, narrowed: list, narrow_contours_info: list):
    self.image = image
    self.narrowed = narrowed
    self.narrow_contours_info = narrow_contours_info
    self.prep = Preprocess()
    self.ocr = paddleOCR()
    self.narrowed_2, self.narrow_contours_info_2, self.areas_2, self.image2 = self.get_sub_narrowed_data()
    print('self.narrowed_2: ', len(self.narrowed_2))

  def get_sub_narrowed_data(self):
    x,y,w,h = self.narrow_contours_info[2]
    image2 = self.image[y:y+h, x:x+w]
    binarized = self.prep.binarize_images([image2], black_on_white=False)
    narrowed_2, narrow_contours_info_2, areas_2 = self.prep.narrow_images(binarized, kernel=(5,5), N=12)
    return narrowed_2[0], narrow_contours_info_2[0], areas_2[0], image2

  def ocr_concat_data(self, ocr_output, discard_checkbox_data = None):
    concatenated_str = ''
    confidence = []
    for data_dict in ocr_output:
      if discard_checkbox_data is None:
        concatenated_str = concatenated_str + data_dict['text'] + ' '
        confidence.append(data_dict['confidence'])
      elif data_dict['text'] != discard_checkbox_data:
        concatenated_str = concatenated_str + data_dict['text'] + ' '
        confidence.append(data_dict['confidence'])
      else:
        pass

    return [concatenated_str, np.mean(confidence)]

  def extract_pertinent_clinical_information(self):
    ocr_output = self.ocr.process(self.narrowed_2[1])
    concatenated_str = self.ocr_concat_data(ocr_output, 'VERBAL')
    return concatenated_str

  def extract_referring_physician_office_stamp(self):
    ocr_output = self.ocr.process(self.narrowed_2[3])
    concatenated_str = self.ocr_concat_data(ocr_output)
    return concatenated_str

  def extract_physician_information(self):
    ocr_output = self.ocr.process(self.narrowed_2[4])
    physician_name = [ocr_output[0]['text'], ocr_output[0]['confidence']]
    date = [ocr_output[1]['text'], ocr_output[1]['confidence']]
    copy_to = [ocr_output[4]['text'], ocr_output[4]['confidence']]
    physician_information = {
                             'physician_name': physician_name,
                             'date': date,
                             'copy_to': copy_to
                             }

    return physician_information

  def extract_ultrasound(self):
    ocr_output = self.ocr.process(self.narrowed_2[5])
    if len(ocr_output) > 1:
      concatenated_str = self.ocr_concat_data(ocr_output[1:])
    else:
      concatenated_str = None
    return concatenated_str

  def extract_xray(self):
    ocr_output = self.ocr.process(self.narrowed_2[7])
    if len(ocr_output) > 1:
      concatenated_str = self.ocr_concat_data(ocr_output[1:])
    else:
      concatenated_str = None
    return concatenated_str

  def extract_appointment_time_date(self):
    # this function is based on theoritical logic as no data is present to test it
    ocr_output = self.ocr.process(self.narrowed_2[9])
    if len(ocr_output) > 1:
      concatenated_str = self.ocr_concat_data(ocr_output[:2]) # :2 because the first two detected objects by OCR will be Time and Date
    else:
      concatenated_str = None
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
    x2,y2,w2,h2 = self.narrow_contours_info_2[6]
    gray = cv2.cvtColor(self.image[y:y+y2,:], cv2.COLOR_RGB2GRAY)
    ocr_output = self.ocr.process(gray)
    patient_info_dict = {
                        "Patient's Last Name": '',
                        "First Name": '',
                        "Sex": '',
                        "Date of Birth": '',
                        "Address": '',
                        "City": '',
                        "Postal Code": '',
                        "Phone Number": '',
                        "Health Card Number": ''
                        }
    health_card_number = ''
    health_confidence = []
    form_labels = list(patient_info_dict.keys())[:-1]
    for i in range(len(ocr_output)):
      similarity_confidence = 0
      data_field = self.detect_patient_information_label_and_associated_data(ocr_output[i])
      if data_field[0] is not None:
        similarity_confidence = self.calc_similarity_confidence(data_field[0], form_labels)
      if isinstance(similarity_confidence, np.ndarray) and similarity_confidence.max() >= 0.5:
        form_label_index = similarity_confidence.argmax()
        detetected_label = form_labels[form_label_index]
        if len(patient_info_dict[detetected_label]) == 0: # if the label has not been assigned a value yet
          patient_info_dict[detetected_label] = [data_field[1], data_field[2]]

      # extracting health card number
      if len(ocr_output[i]["text"]) == 1:
        health_card_number = health_card_number + ocr_output[i]["text"]
        health_confidence.append(ocr_output[i]['confidence'])
      patient_info_dict['Health Card Number'] = [health_card_number, np.mean(health_confidence)]

    """last_name = self.extract_patients_information_associated_data(ocr_output[0])
    address = self.extract_patients_information_associated_data(ocr_output[1])
    city = self.extract_patients_information_associated_data(ocr_output[2])
    first_name = self.extract_patients_information_associated_data(ocr_output[3])
    postal_code = self.extract_patients_information_associated_data(ocr_output[4])
    phone_number = self.extract_patients_information_associated_data(ocr_output[5])
    sex = self.extract_patients_information_associated_data(ocr_output[6])
    dob = self.extract_patients_information_associated_data(ocr_output[7])
    health_card_number_data = ocr_output[9:-1]
    health_card_number = ''
    confidence = []
    for i in range(len(health_card_number_data)):
      health_card_number = health_card_number + health_card_number_data[i]['text']
      confidence.append(health_card_number_data[i]['confidence'])
    health_card_number = [health_card_number, np.mean(confidence)]"""

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
      # Putting a threshold of 150 to detect marked checkboxes
      if w_avg_color <= 150:
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
  
  def insurance(self):
    dict_insurance = {
                        "MSP": False,
                        "ICBC": False,
                        "Private": False,
                        "WorkSafe BC": False,
                        "Other:": False
                        }
    insurance_labels = list(dict_insurance.keys())
    x_i,y_i,w_i,h_i = self.narrow_contours_info_2[6]
    gray_image_crop = cv2.cvtColor(self.image2[y_i:y_i+h_i, x_i:x_i+w_i], cv2.COLOR_RGB2GRAY)
    marked_checkbox_ocr_data = self.vertical_checkboxes_associated_data_extraction(gray_image_crop)
    if marked_checkbox_ocr_data is None:
      return dict_insurance
    else:
      #if len(marked_checkbox_ocr_data) == 1:
      extracted_label = marked_checkbox_ocr_data[0]['text']
      similarity_confidence = self.calc_similarity_confidence(extracted_label, insurance_labels)
      index_of_detected_label = similarity_confidence.argmax()
      detected_label =  insurance_labels[index_of_detected_label]
      if detected_label != "Other:":
        dict_insurance[detected_label] = True
      else:  # For the case Other is marked and has some text beside it
        #print(marked_checkbox_ocr_data)
        output = ''
        confidence = []
        for i in range(1, len(marked_checkbox_ocr_data)):
          output = output + " " + marked_checkbox_ocr_data[i]['text']
          confidence.append(marked_checkbox_ocr_data[i]['confidence'])
        dict_insurance[detected_label] = [output, np.mean(confidence)]

      return dict_insurance