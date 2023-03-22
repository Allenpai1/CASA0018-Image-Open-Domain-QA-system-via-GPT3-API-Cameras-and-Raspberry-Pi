#----------------------------------------------------#
#   System with image input rather than use cameras.
#   Need to input the image path and name at line 36
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
openai.api_key = "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"

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
    
filename = 'IMG_9353.png'  # Image path
img_path='frames.jpg'      
frcnn = FRCNN()
crop = True
count = False
image = Image.open(filename)
#r_image = frcnn.detect_image(image, crop = crop, count = count)
label,r_image = frcnn.detect_image(image, crop = crop, count = count)
label = label_convert(label)
r_image.save(img_path, quality=100, subsampling=0)
image = Image.open(img_path)
image.show()
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
    # if label =='en':
        #myresult= ' '.join(txts)
    #else:
       # myresult= ''.join(txts)
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
    pdf.output("PP.pdf") 
    pages = convert_from_path('PP.pdf', 500)
    for page in pages:
        page.show()
    print('The answer come from GPT are: {}'.format('\n'+'----------------------------------------------------------------------------------------------------------------------------------------------'+
                                                    '\n'+'--->'+'   '+result.strip())+
          '\n'+'----------------------------------------------------------------------------------------------------------------------------------------------')

