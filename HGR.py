import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg=None
def run_avg(image,aw):
    global bg
    if bg is None:
        bg=image.copy().astype("float")
        return
    cv2.accumulateWeighted(image,bg,aw)
def segment(image,threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded=cv2.threshold(diff, threshold,255 ,cv2.THRESH_BINARY)[1]
    #(thresholded)

    (cnts ,_)= cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        #chull=cv2.convexHull(segmented)
        #print(chull.shape())
        return (thresholded, segmented)

def count(thresholded, segmented):
	# find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    #print("chull",chull[:,:])
    #print("chull 1 :",chull[:,:,1])
    # print("chull arg: ",chull[:,:,1].argmin())
    # print("chull chull",chull[chull[:,:,1].argmin()])
    # print("chull fial:",chull[chull[:,:,1].argmin()][0])
    extreme_bottom= tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right= tuple(chull[chull[:, :, 0].argmax()][0])

    cv2.drawContours(clone, [chull], -1, (0, 255, 255), 2)
    cv2.circle(clone, extreme_left, 8, (0, 0, 255), -1)
    cv2.circle(clone, extreme_right, 8, (0, 255, 0), -1)
    cv2.circle(clone, extreme_top, 8, (255, 0, 0), -1)
    cv2.circle(clone, extreme_bottom, 8, (255, 255, 0), -1)


	# find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

	# find the maximum euclidean distance between the center of the palm
	# and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

	# calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

	# find the circumference of the circle
    circumference = (2 * np.pi * radius)

	# take out the circular region of interest which has
	# the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

	# draw the circular ROI

    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    cv2.imshow("mask",circular_roi)
    #cv2.waitKey(0)

	# take bit-wise AND between thresholded hand using the circular ROI as the mask
	# which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
	# compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# initalize the finger count
    count = 0

	# loop through the contours found
    for c in cnts:
		# compute the bounding box of the contour
	    (x, y, w, h) = cv2.boundingRect(c)

		# increment the count of fingers only if -
		# 1. The contour region is not the wrist (bottom area)
		# 2. The number of points along the contour does not exceed
		#     25% of the circumference of the circular ROI
	    if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
		    count += 1

    return count


if __name__=="__main__":
    #initialise weight for the running avg
    aw=0.5
    camera = cv2.VideoCapture(0)
    #region of interest coordinates
    top,right,bottom ,left=40,400,300,650
    nf=0
    while(True):
        (grabbed,frame)=camera.read()
        frame=imutils.resize(frame,width=700)
        frame=cv2.flip(frame,1)
        clone=frame.copy()
        (height, width)= frame.shape[:2]
        roi=frame[top:bottom,right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if nf<30:
            run_avg(gray,aw)
        else:
            hand=segment(gray)
            if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                    fingers = count(thresholded, segmented)
                    if fingers==1:
                        cv2.putText(clone,  ":)    1", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    elif fingers==2:
                        cv2.putText(clone, ":D   2", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    elif fingers==3:
                        cv2.putText(clone, ":O   3", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    elif fingers==4:
                        cv2.putText(clone, ":/    4", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                    else:
                        cv2.putText(clone, ":(   5", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)


                # show the thresholded image
                    cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        nf += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break



# free up memory
camera.release()
cv2.destroyAllWindows()
