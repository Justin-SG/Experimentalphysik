import cv2
import imutils

# Path to video
cap = cv2.VideoCapture('C:/Users/schoe/Desktop/ExPhyTest/20210627_130502.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read video file")

# Path to output file
f = open("C:/Users/schoe/Desktop/ExPhyTest/out.txt", "w")
  
start_frame_number = 12500
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('H','E','V','C'), 120, (frame_width,frame_height))

# Create header line
f.write('Frame;X-Coord;Y-Coord\n')

while True:
  ret, frame = cap.read()

  if ret == True:
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    # Focus are in which the ball is swinging
    ballArea = frame[1050:1450, 190:810]

    # Remove as much useless information as possible to improve tracking
    grayImage = cv2.cvtColor(ballArea, cv2.COLOR_BGR2GRAY)
    blackAndWhiteImage = cv2.inRange(grayImage, 10, 70)
    blackAndWhiteImage = cv2.erode(blackAndWhiteImage, None, iterations=1)
    blackAndWhiteImage = cv2.dilate(blackAndWhiteImage, None, iterations=2)

    
    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(blackAndWhiteImage.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
       # find the largest contour in the mask, then use
       # it to compute the minimum enclosing circle and
       # centroid
       c = max(cnts, key=cv2.contourArea)
       ((x, y), radius) = cv2.minEnclosingCircle(c)
       M = cv2.moments(c)
       center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       # draw the circle and centroid on the frame,
       # then update the list of tracked points
       cv2.circle(blackAndWhiteImage, (int(x), int(y)), int(radius), (120, 120, 0), 2)
       cv2.circle(blackAndWhiteImage, (int(x), int(y)), 1, (120, 120, 0), 5)
       #Output data to file
       f.write('{};{};{}\n'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(x), int(y)))

    # Add frame count to video
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blackAndWhiteImage, str(cap.get(cv2.CAP_PROP_POS_FRAMES)),(0,50), font, 1.5,(255,255,255),2,cv2.LINE_AA)

    # Write frame to file
    out.write(blackAndWhiteImage)
    # Display the resulting frame    
    cv2.imshow('Video', blackAndWhiteImage)
    
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 14000:
        break;

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()
f.close()

# Closes all the frames
cv2.destroyAllWindows()
