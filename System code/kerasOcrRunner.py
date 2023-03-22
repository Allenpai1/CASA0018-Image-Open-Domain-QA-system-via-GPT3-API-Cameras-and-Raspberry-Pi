#----------------------------------------------------#
#   predict single image
#----------------------------------------------------#

import cv2
import matplotlib.pyplot as plt
import keras_ocr
import openai
import math

openai.api_key = "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"

def distinguish_rows(lst, thresh=15):
    """Function to help distinguish unique rows"""
    sublists = []
    for i in range(0, len(lst)-1):
        if (lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh):
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists

def get_distance(predictions):
    """ 
    Function returns dictionary with (key,value):
        * text : detected text in image
        * center_x : center of bounding box (x)
        * center_y : center of bounding box (y)
        * distance_from_origin : hypotenuse
        * distance_y : distance between y and origin (0,0)
    """

    # Point of origin
    x0, y0 = 0, 0 

    # Generate dictionary
    detections = []
    for group in predictions:
    
        # Get center point of bounding box
        top_left_x, top_left_y = group[1][0]
        bottom_right_x, bottom_right_y = group[1][1]
        center_x, center_y = (top_left_x + bottom_right_x)/2, (top_left_y + bottom_right_y)/2

        # Use the Pythagorean Theorem to solve for distance from origin
        distance_from_origin = math.dist([x0,y0], [center_x, center_y])

        # Calculate difference between y and origin to get unique rows
        distance_y = center_y - y0

        # Append all results
        detections.append({
                            'text': group[0],
                            'center_x': center_x,
                            'center_y': center_y,
                            'distance_from_origin': distance_from_origin,
                            'distance_y': distance_y
                        })

    return detections


# keras-ocr will automatically download pretrained
pipeline = keras_ocr.pipeline.Pipeline()
filename = 'frames.jpg'


video = 'http://admin:admin@10.129.196.27:8081/video'  # use for IP camera collection

capture = cv2.VideoCapture(0)  # for iphone use put video instead

#set the width, height and the exposure time
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)


capture.set(cv2.CAP_PROP_EXPOSURE, -8.0)

# button to close window
while capture.isOpened():
    success, img = capture.read()  # read image, one frame by frame
    cv2.imshow("Normal camera",img)
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, blackAndWhiteFrame = cv2.threshold(grayFrame, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('video bw', blackAndWhiteFrame)
    img = cv2.resize(img, (0,0), fx = 0.8, fy = 0.8)  # increase resolution by decrease frames size by 0.8
    frame = cv2.flip(img, flipCode=1)
    key = cv2.waitKey(10)
    if key == 113:  # 'esc' key pressed
        print('Quit the camera window')
        break
    if key == 32: # space key pressed, ascii table
        filename = 'frames.jpg'
        cv2.imwrite(filename, blackAndWhiteFrame)  # save image to current file
        image = keras_ocr.tools.read(filename)
        prediction_groups = pipeline.recognize([image])
        keras_ocr.tools.drawAnnotations(image, prediction_groups[0])
        predictions = prediction_groups[0] # extract text list
        predictions = get_distance(predictions)
        # Set thresh higher for text further apart
        predictions = list(distinguish_rows(predictions, thresh=15))
        # Remove all empty rows
        predictions = list(filter(lambda x:x!=[], predictions))
        ordered_preds = []
        for row in predictions:
            row = sorted(row, key=lambda x:x['distance_from_origin'])
            for each in row: ordered_preds.append(each['text'])
        myresult = ' '.join(ordered_preds)
        print("The text detected are :"+ myresult)
        myresult2 = 'Question detected: ' + myresult
        plt.title(label=myresult2, ha='center', va='bottom', color='red')
        plt.show()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": myresult},
            ]
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content

        print(result)
        
# close camera
capture.release()
