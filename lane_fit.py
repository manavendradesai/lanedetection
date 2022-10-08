#!usr/bin/env python3

from multiprocessing.connection import wait
import cv2 as cv
import numpy as np
import time

# #For grass
# H_low = 0
# H_high = 129
# S_low = 0
# S_high = 19
# V_low = 0
# V_high = 255

#For road
H_low = 0
H_high = 0
S_low = 0
S_high = 0
V_low = 150
V_high = 255

#cap = cv.VideoCapture(0)
#cap = cv.VideoCapture('test1.mov')
#cap = cv.VideoCapture('test3.mov')
cap = cv.VideoCapture('test4.mov')
#cap = cv.VideoCapture('test5.mov')
#cap = cv.VideoCapture('ActualCourseLinesGrass.mov')

########## Detect lanes

while(1):

    # time.sleep(0.05)

    #Take each frame
    ret, frame = cap.read()
    
    #If frame is read correctly ret is True
    #Remove this block of code for online implementation
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #Update HSV values based on trackbars
    lower_white = np.array([H_low,S_low,V_low], np.uint8)
    upper_white = np.array([H_high,S_high,V_high], np.uint8)

    #Mask to get only white colors
    mask = cv.inRange(hsv, lower_white, upper_white)

    #Set top f% of the image to zero
    f = 0.5
    h = len(mask)
    w = len(mask[1]) 
    mask[0:int(np.ceil(f*h)),0:w] = 0

    #Cluster threshold
    ck = w/2

    #Extract rectangles
    er = 3
    kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (er,er))
    rect_mask = cv.morphologyEx(mask, cv.MORPH_OPEN,kernel_rect, iterations=2)

    #Remove sprays
    kernel = np.ones((10,10),np.uint8)
    mask_open = cv.morphologyEx(rect_mask, cv.MORPH_OPEN, kernel)

########## Extract points

    #Draw HoughLinesP
    cdstP = frame.copy()
    #Coarse filtering
    linesP = cv.HoughLinesP(mask_open, 3, 1*np.pi/180, 100, None, 50, 50)

    # # #Filter pixels using lines
    # # mask_endpoints = mask_open.copy()
    # # mask_endpoints[0:h,0:w] = 0

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         #Set endpoints to white
    #         mask_endpoints[l[1], l[0]] = 255
    #         mask_endpoints[l[3], l[2]] = 255

    # #Repeat. Fine filtering
    # # linesP = cv.HoughLinesP(mask_endpoints, 1, 1*np.pi/180, 1, None, 1, 1)
    # #print('Second filter', 2*len(linesP))

    #Filter pixels using lines
    # mask_endpoints = mask_open.copy()
    # mask_endpoints[0:h,0:w] = 0
    
    #Array of lane pixels
    R = np.empty(1,np.int16)
    C = np.empty(1,np.int16)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # cv.line(cdstP, (l[0], l[1]), (l[0], l[1]), (0,0,255), 5, cv.LINE_AA)
            # cv.line(cdstP, (l[2], l[3]), (l[2], l[3]), (0,0,255), 5, cv.LINE_AA)
            # #Set endpoints to white
            # mask_endpoints[l[1], l[0]] = 255
            # mask_endpoints[l[3], l[2]] = 255
            R = np.append(R,[l[1],l[3]])
            C = np.append(C,[l[0],l[2]])

    # R, C = np.where(mask_open==255)
    # print('Rows:', R)
    # print('Columns:',C)

    #Remove random integer in first index
        R = np.delete(R,0)
        C = np.delete(C,0)

    ########## Cluster

        frame_cluster = frame.copy()
        
        #Cluster only if a lane pixel is detected
        if C is not None:

            #Convert to cartesian coordinatess
            x = C    
            y = h-R

            #Initial means
            xm1 = min(w,np.mean(x) + 5)
            ym1 = min(h,np.mean(y) + 5)
            xm2 = max(0,np.mean(x) - 5)
            ym2 = max(0,np.mean(y) - 5)

            #kmeans. Repeat five times
            for i in range(0,10):

                cluster1x = np.empty(1,np.int16)
                cluster1y = np.empty(1,np.int16)
                cluster2x = np.empty(1,np.int16)
                cluster2y = np.empty(1,np.int16)

                #slope of partition
                m = -((ym2-ym1)/(xm2-xm1+0.001)+0.001)**(-1)
                #midpoint 
                xmp = (xm1+xm2)/2
                ymp = (ym1+ym2)/2
                #intercept
                cint = ymp-m*xmp

                for j in range(0,len(y)):
                    if y[j]-m*x[j]-cint<=0:
                        cluster1x = np.append(cluster1x,x[j])
                        cluster1y = np.append(cluster1y,y[j])
                    else:
                        cluster2x = np.append(cluster2x,x[j])
                        cluster2y = np.append(cluster2y,y[j])

                #Eliminate random first entries
                cluster1x = np.delete(cluster1x,0)
                cluster2x = np.delete(cluster2x,0)
                cluster1y = np.delete(cluster1y,0)
                cluster2y = np.delete(cluster2y,0)

                #Revise means
                xm1 = round(sum(cluster1x)/(0.001+len(cluster1x)))
                ym1 = round(sum(cluster1y)/(0.001+len(cluster1y)))
                xm2 = round(sum(cluster2x)/(0.001+len(cluster2x)))
                ym2 = round(sum(cluster2y)/(0.001+len(cluster2y)))

                print('Mean cluster 1:',xm1,ym1)
                print('Mean cluster 2:',xm2,ym2)

            #Image coordinates
            xI1 = cluster1x
            yI1 = h-cluster1y
            nI1 = np.sqrt(xm1**2 + ym1**2)
            xI2 = cluster2x
            yI2 = h-cluster2y
            nI2 = np.sqrt(xm2**2 + ym2**2)

            #Just color conventions
            if nI1<=nI2:
                c1 = (0,0,255)
                c2 = (0,255,0)
            else:
                c2 = (0,0,255)
                c1 = (0,255,0)     

            for i in range(0, len(xI1)):
                cv.line(frame_cluster, (xI1[i], yI1[i]), (xI1[i], yI1[i]), c1, 10, cv.LINE_AA)

            for i in range(0, len(xI2)):
                cv.line(frame_cluster, (xI2[i], yI2[i]), (xI2[i], yI2[i]), c2, 10, cv.LINE_AA)


            #Check for number of lanes
            #x distance between cluster centers
            d = np.abs(xm1-xm2)

            #Detect two lanes if cluster mean separation exceeds image half-width
            if d>ck :
            #Fit two lanes
                #Fit lane 1
                if len(cluster1x)>20:
 
                    lane1 = np.polyfit(yI1,xI1,1)

                    #Start point of lane1
                    ys1 = np.min(yI1)
                    xs1 = round(np.polyval(lane1,ys1))
                    
                    #End point of lane1
                    ye1 = np.max(yI1)
                    xe1 = round(np.polyval(lane1,ye1))

                    cv.line(frame, (xs1, ys1), (xe1, ye1), (255,0,0), 10, cv.LINE_AA)

                #Fit lane 2
                if len(cluster2x)>20:

                    lane2 = np.polyfit(yI2,xI2,1)

                    #Start point of lane1
                    ys2 = np.min(yI2)
                    xs2 = round(np.polyval(lane2,ys2))
                    
                    #End point of lane1
                    ye2 = np.max(yI2)
                    xe2 = round(np.polyval(lane2,ye2))

                    cv.line(frame, (xs2, ys2), (xe2, ye2), (255,0,0), 10, cv.LINE_AA)

            else:
            #Fit one lane
                xI = np.append(xI1,xI2)
                yI = np.append(yI1,yI2)

                if len(xI)>20:

                    lane = np.polyfit(yI,xI,1)

                    #Start point of lane1
                    ys = np.min(yI)
                    xs = round(np.polyval(lane,ys))
                    
                    #End point of lane1
                    ye = np.max(yI)
                    xe = round(np.polyval(lane,ye))

                    cv.line(frame, (xs, ys), (xe, ye), (255,0,0), 10, cv.LINE_AA)

                

    cv.namedWindow('Original frame', cv.WINDOW_NORMAL)
    cv.imshow('Original frame',frame)

# #     #cv.namedWindow('Endpoints',cv.WINDOW_NORMAL)
# #     #cv.imshow('Endpoints',cdstP)

#     # cv.namedWindow('Endpoints mask', cv.WINDOW_NORMAL)
#     # cv.imshow('Endpoints mask',mask_endpoints)

    cv.namedWindow('Clusters', cv.WINDOW_NORMAL)
    cv.imshow('Clusters',frame_cluster)

#     cv.namedWindow('Mask',cv.WINDOW_NORMAL)
#     cv.imshow('Mask',mask_open)

    

    cv.waitKey(2)

    #for button pressing and changing
    if cv.waitKey(2) & 0xFF == ord('d'):
        break


cap.release()
cv.destroyAllWindows()
