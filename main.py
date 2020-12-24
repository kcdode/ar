import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import skimage.io as skio
from skvideo.io import vread, vwrite
import cv2
import sys

SRC = "in/mybox" # Edit this to set the filename
FIRST_TIME = False # Set to True if you need to select the points on the video

VID = vread(SRC + ".mp4") # Edit this to set the file extension
FRAME0_PTS = SRC + "pts"
FRAME0_PTS_3D = SRC + "3dpts"
OUT_VID = SRC + "-out.mp4"
OUT_PTS_VID = SRC + "-ptsout.mp4"

NUM_PTS = 35 # Edit this to edit the number of keypoints on your video
BOX_SIZE = 12 # NxN pixel box for the trackers

def choose_pts(frame, n):
    """
    plt.ginput wrapper, for selecting the points on Frame 0
    """
    plt.imshow(frame)
    plt.show(block=False)
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return np.array(pts)

def gen3D(pts):
    """
    Given chosen keypoints on the 2D image, present each point to the user and ask them to
    enter the 3D point in the to-be AR object
    """
    pts_3d = []
    for point in pts:
        plt.imshow(VID[0])
        plt.scatter(point[0], point[1], c="r")
        plt.show(block=False)
        inp = input("Enter 3D coords 'x,y,z' (no spaces): ").split(",")
        x, y, z = int(inp[0]), int(inp[1]), int(inp[2])
        pts_3d.append([x, y, z])
        plt.close()
    return np.array(pts_3d)

def genBBoxes(pts, box_size):
    """ 
    For each point in Frame 0, establish bounding box coordinates (size BOX_SIZE x BOX_SIZE).
    A point and its bounding box correspond via array index. 
    """
    bboxes = []
    for point in pts:
        x, y = max(0, int(point[0] - box_size/2)), max(0, int(point[1] - box_size/2))
        bboxes.append((x, y, BOX_SIZE, BOX_SIZE))
    assert len(bboxes) == len(pts), "Every point must have a bbox"
    return np.array(bboxes)

def genTrackers(fr0, pts, bbox):
    """
    Create a cv2 tracker for each bounding box, initialized on Frame 0 of the video.
    """
    trackers = []
    for i in range(len(bbox)):
        tracker = cv2.TrackerCSRT_create() # MedianFlow caused lots of errors and yielded poor tracking results
        ok = tracker.init(fr0, tuple(bbox[i]))
        if not ok:
            print("Tracker initialization error")
            sys.exit(0)
        trackers.append(tracker)
    return trackers

def track(trackers, wait=1):
    """
    Show the trackers tracking each point over the duration of the video
    """
    out = []
    for frame in VID:
        # frame = VID[i]
        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else:
                print("err")
        out.append(frame)
        # plt.imshow(frame)
        # plt.show(block=False)
        # plt.pause(wait)
        # plt.close()
    vwrite(OUT_PTS_VID, out)

def pointsFromBBox(bbox):
    """
    Get (approximately) the location of the point (on the 2D frame) from the bbox. Accepts both
    a single bbox (a list or tuple of integers of length 4) and a list of bboxes
    """
    if len(bbox) == 4 and isinstance(bbox[0], np.int32):
        return np.array([bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2])
    points = []
    for box in bbox:
        points.append([box[0] + box[2]//2, box[1] + box[3]//2])
    return np.array(points)

def allPoints(VID, trackers):
    """
    Given the trackers on VID[0], get each point location from the bounding boxes of every tracker on every frame.
    Each index i of all_pts is the set of all 2D points at frame i of the video. 
    """
    all_pts = []
    for i in range(len(VID)):
        frame = VID[i]
        framepts = []
        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                framepts.append([int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)])
            else:   
                print("Error: frame ", i, "tracker ", tracker, bbox)
        all_pts.append(framepts)
    return np.array(all_pts)

def computeM(pts2d, pts3d):
    """
    For Ah = b, solve for h with least squares. Uses all 2D and 3D points in one frame. 
    lstsq has parameter 'rcond=None' just to supress a numpy warning
    """
    bigA = None
    bigB = None
    for i in range(len(pts2d)):
        x, y, z = pts3d[i][0], pts3d[i][1], pts3d[i][2]
        u, v = pts2d[i][0], pts2d[i][1]
        A = np.array([[x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z],
                      [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]])
        b = np.array([[u], [v]])
        if i == 0:
            bigA, bigB = A, b
            continue
        bigA = np.vstack((bigA, A))
        bigB = np.vstack((bigB, b))

    h = np.array(lstsq(bigA, bigB, rcond=None))[0]
    # return h.reshape((3, 4))
    return np.append(h, 1.).reshape((3, 4))
    

def drawAxis(img, imgpts, origin):
   """
   https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html 
   Draws an axis on IMG, with origin at 2D points on the image ORIGIN
   """
   imgpts = np.int32(imgpts).reshape(-1, 2)
#    print(imgpts)
   corner = origin
   img = cv2.line(img, tuple(corner), tuple(imgpts[0].ravel()), (255,0,0), 5)
   img = cv2.line(img, tuple(corner), tuple(imgpts[1].ravel()), (0,255,0), 5)
   img = cv2.line(img, tuple(corner), tuple(imgpts[2].ravel()), (0,0,255), 5)
   return img

def draw(img, imgpts):
    """
    https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html 
    Draws a cube on IMG with 2D coords IMGPTS
    """
    imgpts = np.int32(imgpts).reshape(-1,2)
    # print(imgpts)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


###################
### Entry Point ###
###################

### Select the keypoints. 
if FIRST_TIME:
    pts0 = choose_pts(VID[0], NUM_PTS)
    np.savetxt(FRAME0_PTS, pts0)
else:
    pts0 = np.loadtxt(FRAME0_PTS)

### Create the 3D plane from the selected points
if FIRST_TIME:
    pts3D = gen3D(pts0)
    np.savetxt(FRAME0_PTS_3D, pts3D)
else:
    pts3D = np.loadtxt(FRAME0_PTS_3D)

### Create bounding boxes
bboxes = genBBoxes(pts0, BOX_SIZE)

### Create trackers
trackers = genTrackers(VID[0], pts0, bboxes)

### Visualize trackers on the video (optional)
### Must be run separately from the AR projection because track() modifies the trackers
# track(trackers)
# sys.exit(0)

### Get tracked points
allpts = allPoints(VID, trackers)


axis = np.float32([[3,0,0, 1], [0,3,0, 1], [0,0,-3, 1]]).T
cube = np.float32([[0,0,1,1], [0,1,1,1],[1,1,1,1],[1,0,1,1],
                   [0,0,2,1],[0,1,2,1],[1,1,2,1],[1,0,2,1]]).T


### For each frame:
### - Compute the homography matrix from the 2D to 3D point correspondence
### - Apply it to the base 3D cube points
### - Overlay cube on image and draw
out_frames = []
SHOW_FRAMES = False
for i in range(len(VID)):
    frame = VID[i]
    H = computeM(allpts[i], pts3D)

    # You should only draw either cubes or axis on a given execution of the program
    # cubeCoords = np.dot(H, cube).T[:, :p2]
    # final_frame = draw(frame, cubeCoords)

    axisCoords = np.dot(H, axis).T[:,:2]
    final_frame = drawAxis(frame, axisCoords, allpts[i][0])

    out_frames.append(final_frame)
    if SHOW_FRAMES and i%5 == 0:
        # Only show every 5th
        plt.imshow(final_frame)
        plt.show(block=False)
        plt.pause(.5)
        plt.close()
        
vwrite(OUT_VID, out_frames)
