import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('scanned_document.png')
image2 = cv2.imread('original_document.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match descriptors using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# find correspondences between keypoint descriptors in two images
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on their distances
matches = sorted(matches, key=lambda x: x.distance)

# Apply ratio test and keep only the top matches
good_matches = matches[:int(len(matches) * 0.75)]

# Draw matches
matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Get matched key points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography matrix
homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Warp perspective to align images
aligned_image = cv2.warpPerspective(image1, homography_matrix, (image2.shape[1], image2.shape[0]))

# Create a figure with two subplots (one row, two columns)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the scanned document image
axs[0].imshow(image2, cmap='gray')
axs[0].set_title('Original Document')

# Plot the original document image
axs[1].imshow(aligned_image, cmap='gray')
axs[1].set_title('Scanned Document after Alignment')

# Display the figure
plt.show()