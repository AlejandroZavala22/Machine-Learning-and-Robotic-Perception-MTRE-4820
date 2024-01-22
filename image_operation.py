### Alejandro Zavala
"""
@author: Alejandro Zavala

This is the The update program with all parts integrated
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

capture = cv2.VideoCapture(0)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Used time to obtain an exact amount of frames
frame_set=[]
start_time=time.time()

video = cv2.VideoWriter('live_camara.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30, (frame_width, frame_height))

while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Can\'t get frame')
        break
        
    video.write(frame)
    
    #I have no idea what this do, I need to remove it later
    xyz = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) 
        
    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27:
        print('Pressed Esc')
        break            
    ##This will end the time count and do a quick math to obtain "elapsed"
    end_time=time.time()
    elapsed = end_time - start_time
    if elapsed > 5.001:
       break

capture.release()
video.release()
cv2.destroyAllWindows()

def print_capture_properties(*args):
    capture = cv2.VideoCapture(*args)
    print('Created capture:', *args)
    print('Frame count:', int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('Frame width:', int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame rate:', capture.get(cv2.CAP_PROP_FPS))
    print('\n')
    
print_capture_properties('live_camara.avi')

##############################################################################
##Opening the same video again
capture = cv2.VideoCapture('live_camara.avi')

while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Reached end of video')
        break

    cv2.imshow('frame', frame)
    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break
cv2.waitKey()
capture.release()
cv2.destroyAllWindows()
###############################################################################
#### trying to change color to 3 videos
# read the input color image
capture = cv2.VideoCapture('live_camara.avi')

while True:
    ret, frame = capture.read()    
    
    
    if not ret:
        print('Reached end of video')
        break
    blue,green,red = cv2.split(frame)

    zeros = np.zeros(blue.shape, np.uint8)
    
    blueBGR = cv2.merge([blue,zeros,zeros])
    greenBGR = cv2.merge([zeros,green,zeros])
    redBGR = cv2.merge([zeros,zeros,red])
    
    cv2.imshow('Original', frame)
    cv2.imshow('Blue Channel', blueBGR)
    cv2.imshow('Green Channel', greenBGR)
    cv2.imshow('Red Channel', redBGR)

    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break

capture.release()
cv2.waitKey()
cv2.destroyAllWindows()
#################################################################################
##THIS IF FOR THE RESIZEING TO 70% OF THE VIDEO CAPTURRED
capture = cv2.VideoCapture('live_camara.avi')
while True:
    ret, frame = capture.read()
    

    
    if not ret:
        print('Reached end of video')
        break
    
    newWide= int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)*0.7)
    newHeith= int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.7)
    newsize= cv2.resize(frame, (newWide, newHeith))

    cv2.imshow('70% Resize frame', newsize)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break

capture.release()
cv2.waitKey()
cv2.destroyAllWindows()
####################################################################################
#THIS IS FOR THE HSV AND THE CHANGE INTO THE VALUE OF 50% INTO THE HSV, THIS WILL DISPLAY BOTH HSV AND BGR WITH THE VALUE OF V 50%
capture = cv2.VideoCapture('live_camara.avi')
while True:
    ret, frame = capture.read()

    if not ret:
        print('Reached end of video')
        break
    #RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('The HSV frame',hsv)
    ###########################################################This is for the 50% V value change in the HSV
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)
    
    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    
    # combine new value channel with original hue and sat channels
    hsv2_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv2_gamma, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Gamma correction 50%.jpg',img_gamma2)
    cv2.imshow('HSV2 with 50%.jpg',hsv2_gamma)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break
capture.release()
cv2.waitKey()
cv2.destroyAllWindows()
#################################################################################
## Historygram part
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default) or "rgb"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())

# Configure VideoCapture class instance for using camera or file input.
if not args.get('file', False):
    capture = cv2.VideoCapture('live_camara.avi')
else:
    capture = cv2.VideoCapture(args['file'])

color = args['color']
bins = args['bins']
resizeWidth = args['width']

# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
else:
    ax.set_title('Histogram (grayscale)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
plt.ion()
plt.show()

# Grab, process, and display video frames. Update plot line object(s).
while True:
    (grabbed, frame) = capture.read()

    if not grabbed:
        break

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
            interpolation=cv2.INTER_AREA)
        
        # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(frame.shape[:2])
    if color == 'rgb':
        cv2.imshow('RGB', frame)
        (b, g, r) = cv2.split(frame)
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        cv2.imshow('Ecualized Grayscale', gray_eq)
        cv2.imshow('Grayscale', gray)
        histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
        histogramEqz = cv2.calcHist([gray_eq], [0], None, [bins], [0, 255]) / numPixels
        lineGray.set_ydata(histogramEqz)
        lineGray.set_ydata(histogram)
    fig.canvas.draw()

    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break
# ure.release()
# destroyAllWindows()
cv2.waitKey()
capture.release()
cv2.destroyAllWindows()


#################################################################################
color = cv2.VideoCapture('live_camara.avi')
newvideo = cv2.VideoWriter('modified_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30, (frame_width, frame_height))

while True:
    ret, frame = color.read()
    
    if not ret:
        print('Reached end of video')
        break
    
    
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    
######This is the correction of the HSV into equalized image video
    # grey = cv2.imread('live_camara.avi')
    cv2.imshow('original grey', grey)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()

    grey_eq = cv2.equalizeHist(grey)
    # color = cv2.imread('live_camara.avi')
    hsv123 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv123[..., 2] = cv2.equalizeHist(hsv123[..., 2])
    color_eq = cv2.cvtColor(hsv123, cv2.COLOR_HSV2BGR)
    
    blur = cv2.GaussianBlur(color_eq,(7,7),0)

    cv2.imshow('original color', frame)
    cv2.imshow('equalized color', color_eq)
    cv2.imshow('Gaussian 7x7 noise removal', blur)
    
    newvideo.write(blur)

    
    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break
color.release()
newvideo.release()
cv2.waitKey()
cv2.destroyAllWindows()
    # cv2.waitKey()
    # cv2.destroyAllWindows()
#####################################################################
capture = cv2.VideoCapture('modified_video.avi')

while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Reached end of video')
        break

    cv2.imshow('modified_video', frame)
    key = cv2.waitKey(30)
    if key == 27:
        print('Pressed Esc')
        break
cv2.waitKey()
capture.release()
print('This is the end of the program, and the last video was of the final requirement that was asked from the project')
cv2.destroyAllWindows()