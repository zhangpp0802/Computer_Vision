import cv2 as cv
import numpy as np
import matplotlib.patches as patches


def find_contours(img):
    #img_contrasted = cv.convertScaleAbs(img, alpha=1.2, beta=70) # add contrast to image, planaria will have more light than they otherwise would. do not get filtered out by Otsu.  
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(imgray,(5,5),0)
    kernel = np.ones((5,5),np.uint8) # set up kernel for open morphology (diluting the image, then eroding it)
    opening = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel) # applying open morphology
    _,thresholded_img = cv.threshold(opening,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours,_ = cv.findContours(thresholded_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_threshold_1 = [contours[i] for i in range(len(contours)) if contours[i].shape[0] >= 200 and contours[i].shape[0] < 1200]
    return contour_threshold_1


def find_head(contour_of_planaria):
    min_x = contour_of_planaria[:,0,0].argmin()   # find max and min x, y vals
    max_x = contour_of_planaria[:,0,0].argmax()
    min_y = contour_of_planaria[:,0,1].argmin()
    max_y = contour_of_planaria[:,0,1].argmax()
    x_dist = contour_of_planaria[max_x,0,0] - contour_of_planaria[min_x,0,0]  # compare max and min x,y vals to find orientation of planaria in picture
    y_dist = contour_of_planaria[max_y,0,1] - contour_of_planaria[min_y,0,1]
    print('*'*10, 'Distances between max and min X and max and min Y','*'*10)
    print('\n')
    print('X distance:', x_dist)
    print('Y distance:', y_dist)
  
    if(x_dist > y_dist):   # if the distance between the max and min x are greater than the distance between the max and min y, the planaria is most likely faced from left to right. Otherwise, most likely faced up and down. 
        bottom_half = contour_of_planaria[max_x:min_x,0]    # gets the bottom half of the contour 
        top_half = np.vstack((contour_of_planaria[min_x:,0], contour_of_planaria[:max_x,0])) # gets the top half of the contour
        orientation = 'horizontal'
    else:
        bottom_half = contour_of_planaria[min_y:-max_y,0]    # gets the bottom half of the contour 
        top_half = np.vstack((contour_of_planaria[max_y:,0], contour_of_planaria[:min_y,0])) # gets the top half of the contour
        orientation = 'vertical'
    print('Orientation of planaria:',orientation)
    print('\n')
    if(len(bottom_half) > len(top_half)): # make sure both halves are the same. if not one is going over to the other contour's half. 
        bottom_half = bottom_half[len(bottom_half)-len(top_half):,:]
    elif(len(top_half) > len(bottom_half)):
        top_half = top_half[len(top_half)-len(bottom_half):,:]
    dists_between_top_and_bottom = []  # find the y distances between the two halves of the contours if it is facing left to right, x distances if up and down. max should be where ears are
    max_dist = 0
    for i in range(1, len(top_half)):           # finding the distances between the top and bottom halves of the planaria. The ears will cause a higher displacement
        dist = bottom_half[i,1] - top_half[-i,1]
        dists_between_top_and_bottom.append(dist)
    dists_between_top_and_bottom = np.array(dists_between_top_and_bottom)
    
    earlier_end_of_planaria = dists_between_top_and_bottom[:60]   # we only want to check the ends of the planaria to determine which side ears are on. don't care about middle vals.
    
    start_of_later_end_of_planaria = len(dists_between_top_and_bottom)-60 # indexing the later end of the distances array, so this is our starting pointt

    later_end_of_planaria = dists_between_top_and_bottom[start_of_later_end_of_planaria:]

    max_index_earlier_end_of_planaria = earlier_end_of_planaria.argmax() # 
    max_index_later_end_of_planaria = later_end_of_planaria.argmax()

    if(earlier_end_of_planaria[max_index_earlier_end_of_planaria] > later_end_of_planaria[max_index_later_end_of_planaria]):
        max_dist = max_index_earlier_end_of_planaria
        start_rect = 0
    else:
        max_dist = start_of_later_end_of_planaria+max_index_later_end_of_planaria
        start_rect = start_of_later_end_of_planaria

    if(orientation == 'horizontal'):
            rect = patches.Rectangle((bottom_half[start_rect,0]-90,bottom_half[start_rect,1]-100),130,130,linewidth=5,edgecolor='b',facecolor='none')
    else:
            rect = patches.Rectangle((top_half[start_rect,0]-20, top_half[start_rect+50,1]),140,140,linewidth=5,edgecolor='b',facecolor='none')
    return rect