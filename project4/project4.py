import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def HarrisDetector(img,k = 0.04):

    '''
    Args:
    
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector

    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
   '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    SobelX = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
    SobelY = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
    Ix = cv2.filter2D(np.float64(gray), -1, SobelX)
    Iy = cv2.filter2D(np.float64(gray), -1, SobelY)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    Ixx = cv2.GaussianBlur(Ixx,(5,5),0)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),0)
    Iyy = cv2.GaussianBlur(Iyy,(5,5),0)
    R = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            A  = np.array([Ixx[i,j],Ixy[i,j], Ixy[i,j], Iyy[i,j]]).reshape(2,2)
            R.append(np.linalg.det(A) - k*(np.trace(A)**2))      
    return np.array(R).reshape((gray.shape[0],gray.shape[1]))    


def SuppressNonMax(Rvals, numPts):
    '''
    Args:
    
    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:

     x: A numpy array of shape (N,) containing x-coordinates of interest points
     y: A numpy array of shape (N,) containing y-coordinates of interest points
     confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
   '''
    l = []
    minRadii = []
    maxR = Rvals.max() * .01
    for i in range(Rvals.shape[0]):
        for j in range(Rvals.shape[1]):
            if(Rvals[i,j] > maxR):
                l.append([i,j,Rvals[i,j]])
    descendingR = sorted(l, key=thirdElement, reverse=True)
    for i in range(1,len(descendingR)-1):
        k = i-1
        radii = []
        while(k >= 0):
            radii.append(math.sqrt(((descendingR[i][0] - descendingR[k][0])**2) + ((descendingR[i][1] - descendingR[k][1])**2)))
            k -= 1
        minRadii.append([descendingR[i][0], descendingR[i][1], min(radii)])
    minRadiiDescending = sorted(minRadii,key=thirdElement,reverse=True)
    return [point[1] for point in minRadiiDescending[:numPts]], [point[0] for point in minRadiiDescending[:numPts]]

def thirdElement(l):
    return l[2]

if __name__ == "__main__":
    img = cv2.imread('testimage.pgm')
    R = HarrisDetector(img)
    x,y,z = SuppressNonMax(R,14)
    for i in range(len(x)):
        print(x[i],y[i],z[i])
