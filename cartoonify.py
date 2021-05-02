import numpy as np
import cv2
import matplotlib.pyplot as plt




input_path = 'Slottet_i_Oslo.jpg'
output_path =  None

# Step 1: Read image from path

# Step 2: Find contours
def findCountours(image):
    pass
    
    
# Step 3: Color quantization 
def ColorQuantization(image, K=4):
    
    Z = image.reshape((-1, 3))  
    
    # Convert image to numpy float32
    
    Z = np.float32(Z)
    
    # Define critera and apply kmeans()
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert to uint8 and apply to original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    
    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return res2
# Step n: Combine images

if __name__ == "__main__":
    
    image = cv2.imread(input_path)
    
    image2 = ColorQuantization(image)
    