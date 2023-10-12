# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:28:53 2023

@author: yishu
"""

from preprocess import Preprocess
from template_1 import Template_1
from template_2 import Template_2
import os
import json

class OCR_Engine:
    def __init__(self, img_paths):
        self.img_paths = img_paths
    
    def process_images(self):
        prep = Preprocess()
        images = prep.pdf_to_image(self.img_paths)
        binarized = prep.binarize_images(images, black_on_white=False)
        narrowed_n, narrow_contours_info_n, areas_n = prep.narrow_images(binarized, kernel=(5,25), N=12)
        templates = prep.detect_template(areas_n)
        return templates, images, narrowed_n, narrow_contours_info_n
    

    def extract_template_1(self, image, narrowed_n, narrow_contours_info_n):
        tmplt_1 = Template_1(image, narrowed_n, narrow_contours_info_n)
        pertinent_clinical_information = tmplt_1.extract_pertinent_clinical_information()
        referring_physician_office_stamp = tmplt_1.extract_referring_physician_office_stamp()
        physician_information = tmplt_1.extract_physician_information()
        ultrasound = tmplt_1.extract_ultrasound()
        xray = tmplt_1.extract_xray()
        appointment_time_date = tmplt_1.extract_appointment_time_date()
        patient_info_dict = tmplt_1.extract_patients_information()
        insurance_dict = tmplt_1.insurance()
        template_1_dict = {
                           "PATIENT INFORMATION": patient_info_dict,
                           "INSURANCE": insurance_dict,
                           "X-RAY (No Appointment)": xray,
                           "ULTRASOUND (By Appointment Only)": ultrasound,
                           "PERTINENT CLINICAL INFORMATION (please specify)": pertinent_clinical_information,
                           "APPOINTMENT TIME AND DATE": appointment_time_date,
                           "PHYSICIAN INFORMATION": physician_information,
                           "REFERRING PHYSICIAN OFFICE STAMP": referring_physician_office_stamp
                          }
        return template_1_dict
    
    def extract_template_2(self, image):
        tmplt_2 = Template_2(image)
        referring_physician_office_stamp = tmplt_2.extract_referring_physician_office_stamp()
        physician_information = tmplt_2.extract_physician_information()
        ultrasound = tmplt_2.extract_ultrasound()
        xray = tmplt_2.extract_xray()
        #appointment_time_date = tmplt_1.extract_appointment_time_date()
        #print('appointment_time_date: ', appointment_time_date)
        patient_info_dict = tmplt_2.extract_patients_information()
        fluoroscopy_dict = tmplt_2.fluoroscopy()
        hysterosalpingogram_dict = tmplt_2.extract_hysterosalpingogram()
        insurance_dict = tmplt_2.insurance()
        pertinent_clinical_information = tmplt_2.pertinent_clinical_information()
        template_2_dict = {
                           "PATIENT INFORMATION": patient_info_dict,
                           "INSURANCE": insurance_dict,
                           "X-RAY (No Appointment)": xray,
                           "ULTRASOUND (By Appointment Only)": ultrasound,
                           "FLUOROSCOPY (By Appointment Only)": fluoroscopy_dict,
                           "HYSTEROSALPINGOGRAM (By Appointment at West 8th Location Only)": 
                               hysterosalpingogram_dict,
                           "PERTINENT CLINICAL INFORMATION (please specify)": pertinent_clinical_information,
                           "PHYSICIAN INFORMATION": physician_information,
                           "REFERRING PHYSICIAN OFFICE STAMP": referring_physician_office_stamp
                          }
        return template_2_dict
    
    def extract_ocr_data(self):
        templates, images, narrowed_n, narrow_contours_info_n = self.process_images()
        ocr_data = []
        for i in range(len(templates)):
            if templates[i] == 'template_1':
                ocr_data.append(self.extract_template_1(images[i], 
                                narrowed_n[i], narrow_contours_info_n[i]))
            elif templates[i] == 'template_2':
                ocr_data.append(self.extract_template_2(images[i]))
            else:
                pass
        
        return ocr_data
        

def main():
    img_paths = ['images/Errored Out/_John  2021-07-0216_36_45.pdf']
    #['images/_ BenjaminS2023-06-2912_11_17.pdf',
    #             'images/2023-06-2912_12_05.pdf', 'images/XRay_Sample.pdf']
    ocr_obj = OCR_Engine(img_paths)
    ocr_output_data = ocr_obj.extract_ocr_data()
    for i in range(len(ocr_output_data)):
        file_name = img_paths[i].split('/')[1].split('.')[0]
        output_path = 'output/output_' + file_name + '.txt'
        if not os.path.exists('output'):
            os.makedirs('output')
        with open(output_path, 'w') as convert_file:
            json_output = json.dumps(ocr_output_data[i],indent=4,
                                     separators=(',', ': ')) #sort_keys=True,
            convert_file.write(json_output)

if __name__ == "__main__":
    main()