import cv2
import numpy as np
from matplotlib import pyplot as plt

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def downsample_image(image, reduce_factor):
    for _ in range(reduce_factor):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

# Load calibration data
# ret = np.load('/content/ret.npy')
# K = np.load('/content/K.npy')
# dist = np.load('/content/dist.npy')

ret = None
K = np.eye(3)  # Identity matrix as a placeholder
dist = np.zeros((5, 1))  # Zeros as a placeholder

# Load images
# NANBARGALEEEE NOTE THE BELOW
# note that load images of same size. If the size are not equal resize it

img_path1 = "C:/Users/Arwin/Pictures/profile1.jpg"
img_path2 = "a.jpg"
img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)

# Undistort images
h, w = img_2.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

# Downsample images
img_1_downsampled = downsample_image(img_1_undistorted, 3)
img_2_downsampled = downsample_image(img_2_undistorted, 3)

"""use the below to resize ther image. dont change or use new variable instead update the variable. UNCOMMENT THE BELOW TO RESIZE"""
img_1_downsampled  = cv2.resize(img_1_downsampled, (960, 540)) 
img_2_downsampled = cv2.resize(img_2_downsampled, (960, 540))

"""THESE LINES ARE USED TO CHECK THE SIZE AND TYPE OF THE IMG"""
print(img_1_downsampled.shape, img_1_downsampled.dtype)
print(img_2_downsampled.shape, img_2_downsampled.dtype)


# Stereo matching parameters
win_size = 5
min_disp = -1
max_disp = 63
num_disp = max_disp - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=5,
    disp12MaxDiff=2,
    P1=8 * 3 * win_size**2,
    P2=32 * 3 * win_size**2
)

# Compute disparity map
print("\nComputing the disparity map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)
plt.imshow(disparity_map, 'gray')
plt.show()

# Generate 3D map
print("\nGenerating the 3D map...")
# focal_length = np.load('/content/FocalLength.npy')
# Use a default or estimated focal length (example value)
focal_length = 1000.0  
Q = np.float32([[1, 0, 0, -w / 2.0],
                [0, -1, 0, h / 2.0],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])
Q2 = np.float32([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, focal_length * 0.05, 0],
                 [0, 0, 0, 1]])
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)

# Get colors for 3D points
colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Output file
output_file = 'reconstructed.ply'
print("\nCreating the output file...\n")
create_output(output_points, output_colors, output_file)
