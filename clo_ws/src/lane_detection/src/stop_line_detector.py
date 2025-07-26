#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class StoplineDetector:
    def __init__(self):
        rospy.init_node('StoplineNode', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/BEV_image",CompressedImage,self.callback)

        self.frame = None
        self.roi = None
        self.mask = None
        self.masked = None
        self.gray = None
        self.blur = None
        self.edges = None
        self.count = 0
        cv2.namedWindow("Raw Image", cv2.WINDOW_NORMAL)

    def callback(self, msg):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        
        height, width = self.frame.shape[:2]
        #print(height,width)
        roi_top = int(height /2) + 20
        roi_bottom = height - 80
        self.roi = self.frame[roi_top:roi_bottom, 100:541]
        hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0,0,131])
        upper_white = np.array([40,171,255])

        self.mask = cv2.inRange(hsv, lower_white, upper_white)

        self.masked = cv2.bitwise_and(self.roi, self.roi, mask=self.mask)

        self.gray = cv2.cvtColor(self.masked, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.gray, (5,5), 0)
        self.edges = cv2.Canny(self.blur, 50,150)

        lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, threshold=80, minLineLength=180, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                
                
                if abs(angle) < 2:
                    if (length > 500):
                        continue
                    print(self.count)
                    cv2.line(self.roi, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(self.roi, "Stop Line Detected", (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
                    self.count += 1
                else:
                    cv2.putText(self.roi, "No Stop Line", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.rectangle(self.frame, (100, roi_top), (540, roi_bottom), (0,255,0),2)
        
        
    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Raw Image", self.frame)
                cv2.imshow("ROI", self.roi)
                cv2.imshow("Mask", self.mask)
                cv2.imshow("Masked", self.masked)
                cv2.imshow("gray", self.gray)
                cv2.imshow("blur", self.blur)
                cv2.imshow("edges", self.edges)

                cv2.waitKey(1)
            rate.sleep()

if __name__ == "__main__":
    viewer = StoplineDetector()
    viewer.spin()
    cv2.destroyAllWindows()