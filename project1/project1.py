#project1.py
import numpy as np
import matplotlib.pyplot as plt
import cv2



def loadppm(filename):
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file 
    output: a properly formatted 3d numpy array containing a separate 2d array 
            for each color
    notes: be sure you test for the correct P3 header and use the dimensions and depth 
            data from the header
            your code should also discard comment lines that begin with #
    '''
    with open(filename,"r") as f:
        content = f.readlines()
    del content[0]
    width = int(content[0].strip("\n").split(" ")[0])
    height = int(content[0].strip("\n").split(" ")[1])
    max_color = int(content[1].strip("\n"))
    del content[0]
    del content[0]
    n_list = np.zeros([height,width,3],dtype = "uint8")
    big_mess = []
    for item in content:
        item = item.strip()
        mess_list = item.split(" ")
        while '' in mess_list:
            mess_list.remove('')
        for item in mess_list:
            big_mess.append(item)
    number_list = []
    for item in big_mess:
        number_list.append(int(item))
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            n_list[i,j,0]=number_list[p]
            p +=1
            n_list[i,j,1]=number_list[p]
            p +=1
            n_list[i,j,2]=number_list[p]
            p +=1
            j+=1
        i+=1
    return n_list

def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    height = len(img)
    width = len(img[0])
    g_list = np.zeros([height,width],dtype = "uint8")
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            g_list[i,j] = img[i,j,1]
            j+=1
        i+=1
    return g_list
            

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    height = len(img)
    width = len(img[0])
    b_list = np.zeros([height,width],dtype = "uint8")
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            b_list[i,j] = img[i,j,2]
            j+=1
        i+=1
    return b_list

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    height = len(img)
    width = len(img[0])
    r_list = np.zeros([height,width],dtype = "uint8")
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            r_list[i,j] = img[i,j,0]
            j+=1
        i+=1
    return r_list

def GetGrey(img):
    """to get the greyscale list"""
    height = len(img)
    width = len(img[0])
    grey_list = np.zeros([height,width],dtype = "uint8")
    
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
            grey_list[i,j] = int(r/3+g/3+b/3)
            j+=1
        i+=1
    return grey_list
    
def Threshold(img):
    """get black white image"""
    height = len(img)
    width = len(img[0])
    grey_list = GetGrey(img)
    thre_list = np.zeros([height,width],dtype = "uint8")
    
    i = 0
    p = 0
    while i<height:
        j = 0
        while j < width:
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
            if grey_list[i,j]<128:
                thre_list[i,j] = 0
            else:
                thre_list[i,j] = 255
            j+=1
        i+=1
    return thre_list
    
# def Histogram(img):
#     """make histogram image"""
#     height = len(img)
#     width = len(img[0])
#     grey_list = np.zeros([height,width],dtype = "uint8")
    
#     i = 0
#     p = 0
#     while i<height:
#         j = 0
#         while j < width:
#             r = img[i,j,0]
#             g = img[i,j,1]
#             b = img[i,j,2]
#             grey_list[i,j] = int(r/3+g/3+b/3)
#             j+=1
#         i+=1
#     print(grey_list)
#     return grey_list
    
def Histogram(img):
    """make histogram image"""
    height = len(img)
    width = len(img[0])
    grey_list = GetGrey(img)
    print(grey_list)
    i = 0
    
    mess_list = []
    while i<height:
        j = 0
        while j < width:
            grey = grey_list[i,j]
            mess_list.append(grey)
            j+=1
        i+=1
    
    c_list = []
    for i in range (0,256):
        c_list.append(mess_list.count(i))
    print(c_list)

    cc_list = c_list[:]
    
    cdfx = []
    pixels = height*width
    for i in range (0,len(c_list)):
        if i > 0:
            cc_list[i] = cc_list[i-1] + (c_list[i])
            #print(c_list[i])
            cdfx.append(cc_list[i])
        else:
            cc_list[i] = (c_list[i])
            cdfx.append(cc_list[i])
    print(cdfx)
    h_list = []
    min0 = cdfx[0]
    pixels = height*width
    ah = pixels-min0
    for i in range (0,len(cdfx)):
        h = int((((cdfx[i]-min0)*1.0)/(ah*1.0))*255)
        h_list.append(h)
    print(h_list)
        
    grei_list = np.zeros([height,width],dtype = "uint8")
    i = 0
    while i < height:
        j = 0
        while j < width:
            grei_list[i,j] = h_list[grey_list[i,j]]
            j+=1
        i+=1
    return grei_list
    
    
    
    
  
    
    
    
    
       
   
    
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == "__main__":
  #put any command-line testing code you want here.
  pass
