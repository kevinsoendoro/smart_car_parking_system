# Resolution 1280x720
import cv2
import numpy as np
import keyboard as keys
import time
object_detector = cv2.createBackgroundSubtractorMOG2(history = 150, varThreshold = 60, detectShadows = False)
video = cv2.VideoCapture('video_parking_2x.mp4')
cnt = 0
while(1):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale", gray)
    # frame2 = copy.copy(frame)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # cv2.imshow("blur", blur)
    # Edge Detection
    edges = cv2.Canny(blur, 150, 150, apertureSize=3)
    cv2.rectangle(edges, (0,0), (1280, 215), (0,0,0), -1)
    pt1 = (0, 196)
    pt2 = (0, 720)
    pt3 = (156, 720)
    pt4 = (213, 199)
    triangle_cnt = np.array([pt1, pt2, pt3, pt4])
    cv2.drawContours(edges, [triangle_cnt], 0, (0,0,0), -1)
    pt5 = (1070, 178)
    pt6 = (1280, 178)
    pt7 = (1280, 720)
    pt8 = (1132, 720)
    triangle_cnt2 = np.array([pt5, pt6, pt7, pt8])
    cv2.drawContours(edges, [triangle_cnt2], 0, (0,0,0), -1)
    pt9 = (0, 1280)
    pt10 = (1280, 720)
    pt11 = (1280, 626)
    pt12 = (0, 647)
    triangle_cnt3 = np.array([pt9, pt10, pt11, pt12])
    cv2.drawContours(edges, [triangle_cnt3], 0, (0,0,0), -1)
    cv2.imshow("Edges", edges)

    # Region of Interest
    #area_1 = [(270, 400), (350, 400), (350, 450), (270, 450)]
    #area_2 = [(470, 400), (550, 400), (550, 450), (470, 450)]
    area_3 = [(601, 236), (838, 205), (917, 662), (621, 673)]
    #area_4 = [(660, 400), (777, 400), (777, 450), (660, 450)]
    #area_5 = [(0,0), (1280, 0), (1280, 720), (0, 720)]

    # Area Available Test
    for area in [area_3]:
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 4, -1)
    mask = object_detector.apply(frame)
    kernel  = np.ones((8,8), np.uint8)
    erosion = cv2.erode(frame ,kernel,iterations = 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 6000:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = int(x+w)
            cy = int(y+h)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            result = cv2.pointPolygonTest(np.array(area_3, np.int32), (int(cx), int(cy)), False)
    if result >= 0 :
        cv2.putText(frame,"3 Terisi", (471, 120), cv2.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0), 4)
        print("3 Terisi")
    #cv2.imshow("Back Sub", mask)
    # cv2.imshow("Edge Detection", edges)
    
    lines = cv2.HoughLinesP(
        edges, # Input edge image
        1, # Distance resolution in pixels
        np.pi/180, # Angle resolution in radians
        threshold=75, # Min number of votes for valid line
        minLineLength=20, # Min allowed length of line
        maxLineGap=10 # Max allowed gap between line for joining them
        )
    for points in lines:
        x1, x2, y1, y2 = points[0]
        cv2.line(frame, (x1, x2), (y1, y2), (0, 255 , 0), 6)
        if x1 > 1:
            cx2 = int(x1+x2)
            cy2 = int(y1+y2)
            arealine = [(610, 259), (633, 254), (659, 657), (626, 658)]
            arealine2 =[(804, 238), (855, 650), (886, 644), (826, 229)]
            cv2.polylines(frame, [np.array(arealine, np.int32)], True, (0, 255, 0), 4, -1)
            cv2.polylines(frame, [np.array(arealine2, np.int32)], True, (0, 255, 0), 4, -1)
            result2 = cv2.pointPolygonTest(np.array(arealine, np.int32), (int(cx2), int(cy2)), False)
            result3 = cv2.pointPolygonTest(np.array(arealine2, np.int32), (int(cx2), int(cy2)), False)
    # Slot 1
    cv2.circle(frame, (331,189), 20, (0,0,0), -1)
    # Slot 2
    cv2.circle(frame, (530,189), 20, (0,0,0), -1)
    # Slot 3
    cv2.circle(frame, (713,189), 20, (0,0,0), -1)
    if result2 >= 0 and result >= 0:
        cv2.circle(frame, (713,189), 20, (0,255,0), -1)
    # Slot 4
    cv2.circle(frame, (920,189), 20, (0,0,0), -1)

    # Nomor Slot
    cv2.putText(frame,"1", (250, 670), cv2.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0), 4)
    cv2.putText(frame,"2", (450, 670), cv2.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0), 4)
    cv2.putText(frame,"3", (710, 670), cv2.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0), 4)
    cv2.putText(frame,"4", (930, 670), cv2.FONT_HERSHEY_SIMPLEX, 2 , (255,0,0), 4)
    cv2.imshow('Real Video', frame)

    key = cv2.waitKey(1)
    if (key==27):
        break

video.release()
cv2.destroyAllWindows()