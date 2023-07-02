import cv2
import numpy as np
import matplotlib.pyplot as plt

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
# The weight of each image decreases with its index to create the fade effect
for i, red_circle in enumerate(red_circles):
    weight = 1 - i / num_images  # Decreases from 1 to 0 as i increases
    final_image = cv2.addWeighted(final_image, 1, red_circle, weight, 0)

# Convert final image to RGB for matplotlib
final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

# Use plt.show to display the image
plt.imshow(final_image)
plt.axis('off')  # Hide axes
plt.show()

# Save the final image
cv2.imwrite('final.png', final_image)
