import cv2
import numpy as np
import skimage.feature
import os

def data_extraction(name:str, dataset:str):
    src = f'Skin Cancer/{name}'

    color = cv2.imread(src,-1)
    gray = cv2.imread(src,0)
    v = cv2.THRESH_BINARY_INV
    if all(color[0][0] == [0]):
        gray = gray[45:395, 50:560]
        color = color[45:395, 50:560]
        v = cv2.THRESH_BINARY

    # Preprocesing
    # RazorDull Method to remove hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) 
    noiseless_col = cv2.morphologyEx(gray,  cv2.MORPH_BLACKHAT, kernel) 
    mask = cv2.threshold(noiseless_col, 20, 255, cv2.THRESH_BINARY)[1]
    noiseless_col = cv2.inpaint(color,mask,6,cv2.INPAINT_TELEA)  


    # Noise reduction
    noiseless_col = cv2.medianBlur(noiseless_col, ksize=1)
    noiseless_col = cv2.GaussianBlur(noiseless_col, (11,11), 0)

    # gray image without noise
    noiseless_gray = cv2.cvtColor(noiseless_col, cv2.COLOR_BGR2GRAY)

    # Segmentation 
    pixel_vals = noiseless_gray.reshape((-1,1))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    k = 3
    labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1::]
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()]

    # segmented image
    segmented_image = segmented_data.reshape((noiseless_gray.shape)) 


    thresh_cntless = cv2.threshold(noiseless_gray, 0, 255, v + cv2.THRESH_OTSU)[1]
    contours,_ = cv2.findContours(thresh_cntless, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cmax = max(contours, key = cv2.contourArea) 
    epsilon = 0.0002 * cv2.arcLength(cmax, True)
    approx = cv2.approxPolyDP(cmax, epsilon, True)
    f = cv2.drawContours(noiseless_col, [approx], -1, (0, 0, 0), 4)
    width, height = noiseless_gray.shape
    thresh = np.zeros( [width, height, 3],dtype=np.uint8 )
    thresh = cv2.fillPoly(thresh, pts =[cmax], color=(255,255,255))


    noiseless_col[thresh==0] = [0]
    noiseless_gray = cv2.cvtColor(noiseless_col, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('q', thresh)
    # cv2.imshow("Filtered", noiseless_col)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    #diameter
    contours,hierarchy = cv2.findContours(noiseless_gray,2,1)
    for i in range (len(contours)):
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        diameter = (int(radius)*2)/100
        
        
    #asymmetry
    top_half = thresh[0:225, :]
    bottom_half = thresh[225:, :]
    left_half = thresh[:, 0:300]
    right_half = thresh[:, 300:]

    top_half = top_half[:, :, 1]
    bottom_half = bottom_half[:, :, 1]
    left_half = left_half[:, :, 1]
    right_half = right_half[:, :, 1]

    try:
        top_bot_ratio = (int(np.size(top_half) - np.count_nonzero(top_half)) / (np.size(bottom_half) - np.count_nonzero(bottom_half)))
    except:
        top_bot_ratio = 1
        
    try:
        left_right_ratio = (int(np.size(left_half) - np.count_nonzero(left_half)) / (np.size(right_half) - np.count_nonzero(right_half)))
    except:
        left_right_ratio = 1
    
        
    asymmetry = (int(top_bot_ratio) + int(left_right_ratio)) / 2
    asymmetry = 1 if asymmetry>1 else asymmetry
    


    #color
    def val(color: list):
        col = list(map(lambda x: sum(x)/len(x), color))
        col = sum(col)/len(col)
        return col
    blue, green, red = cv2.split(noiseless_col)
    blue, green, red = round(val(blue),4), round(val(green),4), round(val(red),4)


    co_matrix = skimage.feature.graycomatrix(noiseless_gray, [5], [0], levels=256, symmetric=True if asymmetry!=2 else False, normed=True)

    # Calculate texture features from the co-occurrence matrix
    contrast = round(skimage.feature.graycoprops(co_matrix, 'contrast')[0][0],4)
    correlation = round(skimage.feature.graycoprops(co_matrix, 'correlation')[0][0],4)
    energy = round(skimage.feature.graycoprops(co_matrix, 'energy')[0][0],4)
    homogeneity = round(skimage.feature.graycoprops(co_matrix, 'homogeneity')[0][0],4)
    
    if not os.path.isfile(f'{dataset}.csv'):
        with open(f'{dataset}.csv', 'a') as f:
            f.write(f'name,asymmetry,diameter,red,green,blue,contrast,correlation,energy,homogeneity\n')

    with open(f'{dataset}.csv', 'a') as f:
        f.write(f'{name[0:12]},{asymmetry},{diameter},{red},{green},{blue},{contrast},{correlation},{energy},{homogeneity}\n')
    return 0


def data_load(i:int, j:int, dataset:str):
    l = os.listdir('Skin Cancer')[i:j] if j != 0 else os.listdir('Skin Cancer')[i::]
    g = list(map(lambda x:data_extraction(name=x, dataset=dataset), l))
    

data_load(0, 0, 'fin')