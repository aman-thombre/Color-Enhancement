import cv2
import numpy as np

def histograms(im):
    
    '''
    Returns color histograms of images. 

    INPUTS:
        im: input image

    OUTPUTS:
        histograms: color histograms of images, counts of each color in an image
        cumsum: cumulative sum of these histograms, used in histogram equalization
    '''
    histograms = np.zeros((3,256))
    cumsum = np.zeros((3,256))
    
    for color in range(3):
        for i in range(256):
            histograms[color,i] = len(im[im[:,:,color] == i]) # count frequency of intensity at each color
        cumsum[color,:] = np.cumsum(histograms[color,:]) # get cumsum of these histograms
        
    return histograms, cumsum

def hist_eq(im, alpha):
    
    '''
    Performs histogram equalization of input image for color enhancement. 

    INPUTS:
        im: input image
        alpha: weighting factor

    OUTPUTS:
        im_new: new, histogram equalized image
    '''

    im_new = np.zeros(np.shape(im))
    rows, columns, colors = np.shape(im)

    _, cumsum = histograms(im)

    for color in range(colors):
        for r in range(rows):
            for c in range(columns): 
                i = int(im[r, c, color]) # grab intensity at a location
                im_new[r,c,color] = int(alpha * cumsum[color,i] / cumsum[color,-1] * 255 + (1-alpha) * i) # new location is weighted sum of equalized intensity + original intensity

    return im_new

def gamma_correct(im, gamma):
    '''
    Performs gamma correction of saturation of input image for color enhancement. 

    INPUTS:
        im: input image
        gamma: gamma correction factor - if < 1, then boosts lower intensities more, if > 1, boosts higher intensities more

    OUTPUTS:
        im_new: enahnced image
    '''
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV) # convert to HSV color space
    im[:,:,1] **= gamma # gamma correction
    im_new = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)

    return im_new