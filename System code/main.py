#----------------------------------------------------#
#   Main program with camera connection
#   Change the openai.api_key to you own
#   And video ip address, currently is connected with computer camera
#----------------------------------------------------#

from frcnn import FRCNN
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import openai
import csv
import time
import numpy as np
from frcnn import FRCNN
from fpdf import FPDF
from pdf2image import convert_from_path
from PIL import Image

def label_convert(lab):
    if lab == 0:
        return 'ch'
    elif lab == 'nan' :
        print('The model language detection may be inaccurate')
        return 'en'
    else:
        return 'en'
    
def real_label_convert(lab):
    if lab == 'en':
        return 'English'
    elif lab == 'ch' :
        return 'Chinese'

openai.api_key = "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"
#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
filename = 'frames.jpg'

video = 'http://admin:admin@192.168.0.15:8081/video'  # use for IP camera collection

capture = cv2.VideoCapture(0)  # for iphone use put video instead

#set the width, height and the exposure time
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

capture.set(cv2.CAP_PROP_EXPOSURE, -8.0)

# button to close window
while capture.isOpened():
    success, img = capture.read()  # read image, one frame by frame
    cv2.imshow("Normal camera",img)
    img = cv2.resize(img, (0,0), fx = 0.8, fy = 0.8)  # increase resolution by decrease frames size by 0.8
    frame = cv2.flip(img, flipCode=1)
    key = cv2.waitKey(10)
    if key == 113:  # 'esc' key pressed
        print('Quit the camera window')
        break
    if key == 32: # space key pressed, ascii table
        filename = 'frames.jpg'
        cv2.imwrite(filename, img)  # save image to current file
        img_path = "frames.jpg"
        frcnn = FRCNN()
        crop = True
        count = False
        image = Image.open(filename)
        #r_image = frcnn.detect_image(image, crop = crop, count = count)
        label,r_image = frcnn.detect_image(image, crop = crop, count = count)
        label = label_convert(label)
        #r_image.save(img_path, quality=100, subsampling=0)
        #image = Image.open(img_path)
        r_image.show()
        ocr = PaddleOCR(use_angle_cls=True, lang=label) 
        result = ocr.ocr(img_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
            result = result[0]
            image = Image.open(img_path).convert('RGB')
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
            im_show = Image.fromarray(im_show)
            im_show.show()
            w = 190
            h = 10
            pdf = FPDF()  
            pdf.add_page()
            pdf.set_font("Arial", size = 15) 
            pdf.add_font('sysfont', '', r"fireflysung.ttf", uni=True)
            pdf.set_font('sysfont', '', 12)
            myresult= ' '.join(txts)
            pdf.cell(200, 10, txt = "",  ln = 2)
            pdf.multi_cell(w, h, 'Question: \n'+myresult+'?', border=1)
            print("The language detected is: {}".format('\n'+'--------------------------------------------------------------------------------------------------------'+
                                                        '\n'+'|'+'                                            '+real_label_convert(label)+ '                                                   ' + '|' +
                                                        '\n'+'--------------------------------------------------------------------------------------------------------'))
            print('The question detected is: {}'.format('\n'+'--------------------------------------------------------------------------------------------------------'+
                                                        '\n'+'--->'+'   '+myresult+'\n'+'--------------------------------------------------------------------------------------------------------'))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": myresult},
                ]
            )
            result = ''
            for choice in response.choices:
                result += choice.message.content
                print(str(choice.message.content).strip())
                pdf.multi_cell(w, h, 'GPT answer: \n'+str(choice.message.content).strip(),border=1)
            pdf.output("QA_results.pdf") 
            pages = convert_from_path('QA_results.pdf', 500)
            for page in pages:
                page.show()
            print('The answer come from GPT are: {}'.format('\n'+'----------------------------------------------------------------------------------------------------------------------------------------------'+
                                                            '\n'+'--->'+'   '+result.strip())+
                '\n'+'----------------------------------------------------------------------------------------------------------------------------------------------')
# close camera
capture.release()