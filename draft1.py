import cv2
import numpy as np

# Total number of images to process
num_images = 10

# Load all images
images = [cv2.imread(f"fig{i+1}.jpeg") for i in range(num_images)]

# Convert the images to HSV color space
hsv_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

# Define color range for the red color in HSV
lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])

# Detect the red circle in each image and create a mask
masks = [cv2.inRange(hsv, lower_red, upper_red) for hsv in hsv_images]

# Bitwise-and mask and original image
red_circles = [cv2.bitwise_and(img, img, mask=mask) for img, mask in zip(images, masks)]

# Now create an output image which is completely black
final_image = np.zeros_like(images[0])

# Add up all images containing the red circles
for red_circle in red_circles:
    final_image = cv2.addWeighted(final_image, 1, red_circle, 1, 0)

# Show the final image
cv2.imshow("Stroboscopic Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image
cv2.imwrite('final.png', final_image)
