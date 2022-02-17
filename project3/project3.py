import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def myHoughLines(image, rho, theta, threshold):
    thetaVals = np.linspace(0, math.pi/2, round(math.pi/theta))
    rhoMax = np.hypot(image.shape[0],image.shape[1])
    rhoMaxIndex = int(round(rhoMax/rho))
    accum = np.zeros((rhoMaxIndex, thetaVals.shape[0]))
    out = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j,0] == 255):
                for t in range(thetaVals.shape[0]):
                    r = j*math.cos(thetaVals[t]) + i*math.sin(thetaVals[t])
                    accum[round(r/rho), t] += 1
    for i in range(accum.shape[0]):
        for j in range(accum.shape[1]):
            if accum[i,j] > threshold:
                out.append((i*rho,thetaVals[j]))
    return out

def nonMaxSuppression(mag, angle):
    magPad = np.pad(mag, ((1,1),(1,1),(0,0)), 'constant', constant_values=0)
    out = np.copy(mag)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            comp1 = 0
            comp2 = 0
            if angle[i,j, 0] <= 22.5 or angle[i,j, 0] > 157.5:      # if theta is closest 0/180 degrees, check left and right neighbors
                comp1 = magPad[i+1,j]
                comp2 = magPad[i+1,j+2]
            elif angle[i,j, 0] > 22.5 and angle[i,j, 0] <= 67.5:    # if theta is closest to 45 degrees, check NE and SW neighbors 
                comp1 = magPad[i,j+2]
                comp2 = magPad[i+2,j]
            elif angle[i,j, 0] > 67.5 and angle[i,j, 0] <= 112.5:   # if theta is closest to 90 degrees, check top and bottom neighbors
                comp1 = magPad[i, j+1]
                comp2 = magPad[i+2, j+1]
            elif angle[i,j, 0] > 112.5 and angle[i,j, 0] <= 157.5:  # if theta is closest to 135 degrees check NW and SE neighbors
                comp1 = magPad[i,j]
                comp2 = magPad[i+2, j+2]
            if((mag[i,j,0] >= comp1[0]) and (mag[i,j,0] >= comp2[0])):
                out[i,j] = mag[i,j,0]
            else:
                out[i,j] = 0
    return out


def DoubleThreshold(img, Tlow, Thi):
    weak = 10
    strong = 255
    discard = 0
    out = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(out[i,j,0] < Tlow):
                out[i,j] = discard
            elif(out[i,j,0] >= Tlow and out[i,j,0] < Thi):
                out[i,j] = weak
            elif(out[i,j,0] >= Thi):
                out[i,j] = strong
    return np.uint8(out)

def EdgeTracking(img):
    weak = 10
    strong = 255
    discard = 0
    imgPad = np.pad(img, ((1,1),(1,1),(0,0)), 'constant', constant_values=0)
    out = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(out[i,j,0] == weak):
                if((imgPad[i+1,j,0] == strong) or (imgPad[i+1,j+2,0] == strong) or (imgPad[i,j+2,0] == strong) or
                   (imgPad[i+2,j,0] == strong) or (imgPad[i, j+1,0] == strong) or (imgPad[i+2, j+1,0] == strong) or
                   (imgPad[i,j,0] == strong) or (imgPad[i+2, j+2,0] == strong)):
                    out[i,j] = strong
                else:
                    out[i,j] = discard
    return out


def sobelThreshold(img, threshold):
    newImg = img.copy()
    newImg[img < threshold] = 0
    newImg[img >= threshold] = 255
    return newImg

def Canny(image, threshold1, threshold2):
    mag,angle = np.dsplit(MagnitudeAndAngle(image),2)
    newMag = nonMaxSuppression(mag, angle)
    doubleThres = DoubleThreshold(newMag, threshold1, threshold2)
    return EdgeTracking(doubleThres)

def MagnitudeAndAngle(image):
    filter = np.ones((5, 5))
    filter /= np.sum(filter)
    img64 = np.array(image, np.float64)
    blurred64 = cv2.filter2D(img64, -1, filter)
    SobelX = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
    SobelY = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
    Gx = cv2.filter2D(blurred64, -1, SobelX)
    Gy = cv2.filter2D(blurred64, -1, SobelY)
    mag = np.sqrt(np.square(Gx) + np.square(Gy))
    mag_norm = mag / np.max(mag)
    theta = np.arctan2(Gy, Gx)
    theta = theta*180/math.pi
    theta = 180 - theta
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            for k in range(mag.shape[2]):
                if(theta[i,j,k] < 0):
                    theta[i,j,k] += 180

    return np.dstack((mag_norm,theta))



if __name__ == '__main__':
    #TESTING CODE HERE

    # Edge Detecting Filters
    Gx = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
    Gy = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
    # Small Blur Filter
    blur_filter = np.ones((5, 5))
    blur_filter /= np.sum(blur_filter)  # making the filter sum to 1
    stripe = np.float64(np.array(cv2.imread('../project3/images/stripe.pgm')))
    blurred = cv2.GaussianBlur(stripe,(5,5),0)
    vert_edge = cv2.filter2D(blurred, -1, Gx)
    horiz_edge = cv2.filter2D(blurred, -1, Gy)
    mag = (np.sqrt(np.square(vert_edge) + np.square(horiz_edge)))
    newMag = sobelThreshold(mag, 100)
    lines = myHoughLines(newMag,20,math.pi/10,200)
    t = [l[1] for l in lines]
    r = [l[0] for l in lines]
    plt.plot(t,r)
    plt.scatter(t,r)
    plt.show()
    