from PIL import Image
import numpy as np
import scipy.ndimage
from mlbase import network as N


# task here:
# 1. eye window, for a input image, the focus has high resolution and low resolution at other place.
# 2. move the image is the same as move the high resolution window.
# 3. eye should move the focus window to trakcing the same focus.
# 4. training is around the deep network. the input is the current eyed image, with high/low resolution region.
# 5. the output of the network should be the action to take to move the eye
# 6. the loss of the network should be activation update energy cost?
# 7. the network should has different pooling stride size for the high/low resolution region to compensate density of vision neuron.



def drawRectangle(img, left, top, bottom, right):
    img[0, left:right, top:bottom] = 1
    img[1, left:right, top:bottom] = 0
    img[2, left:right, top:bottom] = 0
    return img

def drawBBox(img, left, top, bottom, right):
    """
    drawBBox(img, left, top, bottom, right)
    """
    halflinewidth = 2
    img = drawRectangle(img, left-halflinewidth, top, bottom, left+halflinewidth)
    img = drawRectangle(img, left, top-halflinewidth, top+halflinewidth, right)
    img = drawRectangle(img, right-halflinewidth, top, bottom, right+halflinewidth)
    img = drawRectangle(img, left, bottom-halflinewidth, bottom+halflinewidth, right)
    return img

def readImage(imgpath):
    i = Image.open(imgpath)
    a = np.asarray(i)
    b = np.array(a)
    c = np.rollaxis(b, 2, 0)
    c = c/255
    return c

def back2PIL(a):
    a3 = np.moveaxis(a, 0, 2)
    b3 = a3*255
    c3 = b3.astype(np.uint8)
    img = Image.fromarray(c3)
    return img

def eyeWindow(img, center, radius, density_ratio):
    (channel, width, height) = img.shape
    out = np.array(img)
    out = scipy.ndimage.zoom(out, (1, 1/density_ratio, 1/density_ratio))
    out[out<0] = 0
    out[out>1] = 1
    out = scipy.ndimage.zoom(out, (1, density_ratio, density_ratio))
    out[out<0] = 0
    out[out>1] = 1
    out[:, int(center[0]-radius):int(center[0]+radius), int(center[1]-radius):int(center[1]+radius)] = img[:, int(center[0]-radius):int(center[0]+radius), int(center[1]-radius):int(center[1]+radius)]
    
    return out

    
def trackingNetwork():
    n = N.Network()
    return n
    

imagef = '/hdd/home/yueguan/workspace/sesame-paste-noodle-dev/examples/Large_Pinus_glabra.jpg'
    out = scipy.ndimage.zoom(out, (1, density_ratio, density_ratio))
    out[:, int(center[0]-radius):int(center[0]+radius), int(center[1]-radius):int(center[1]+radius)] = img[:, int(center[0]-radius):int(center[0]+radius), int(center[1]-radius):int(center[1]+radius)]
    
    return out
    

    

imagef = '/Users/yguan/workspace/sesame-paste-noodle/examples/Large_Pinus_glabra.jpg'


img = readImage(imagef)
img = eyeWindow(img, (1000, 1000), 200, 16)
img = drawBBox(img, 800, 800, 1200, 1200)
pilimg = back2PIL(img)


