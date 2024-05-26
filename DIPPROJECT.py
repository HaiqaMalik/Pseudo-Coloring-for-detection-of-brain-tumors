import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

def load_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Original Image", original_image)
        cv2.waitKey(3000)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray_image, 50, 50)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        black_image = np.zeros_like(image)
        mask = cv2.drawContours(black_image, [largest_contour], -1, (255, 255, 255), cv2.FILLED)
        
        cropped_image = cv2.bitwise_and(image, cv2.bitwise_not(mask))
        #cv2.imshow('Cropped Image', cropped_image)

        # Binarization
        _, binarized_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binary Image", binarized_image)
        cv2.waitKey(3000)

        equalized_image = equalization(binarized_image)
        #cv2.imshow("Equalized Image", equalized_image)
        cv2.waitKey(3000)

        smoothed_image = gaussian_blur(equalized_image, kernel_size=5, sigma=1)
        #cv2.imshow("smoothed Image", smoothed_image)
        cv2.waitKey(3000)

        sliced_image = density_slicing(image)
        cv2.imshow("Sliced Image", sliced_image)
        cv2.waitKey(3000)

        segmented_image = segmentation_yCbCr(sliced_image)
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

def equalization(binarized_image):
    histogram = np.zeros(256)
    for i in range(binarized_image.shape[0]):
        for j in range(binarized_image.shape[1]):
            histogram[binarized_image[i, j]] += 1

    scale_factor = 255.0 / (binarized_image.shape[0] * binarized_image.shape[1])

    equalized_histogram = np.zeros(256)
    equalized_histogram[0] = scale_factor * histogram[0]
    for i in range(1, 256):
        equalized_histogram[i] = equalized_histogram[i - 1] + scale_factor * histogram[i]

    equalized_histogram = np.clip(equalized_histogram, 0, 255)

    equalized_image = np.zeros_like(binarized_image)
    for i in range(binarized_image.shape[0]):
        for j in range(binarized_image.shape[1]):
            equalized_image[i, j] = np.round(equalized_histogram[binarized_image[i, j]]).astype(np.uint8)
    return equalized_image
def gaussian_blur(image, kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2*sigma**2)), (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)

    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_width=pad_size, mode='reflect')
    
    convolved_image = np.zeros_like(image, dtype=np.float32)
    convolved_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                convolved_image[i, j, k] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size, k] * kernel)

    return convolved_image.astype(np.uint8)        

def density_slicing(image):
    colors = [
        (0, 0, 255),    # Blue for pixel values 0-55
        (0, 255, 0),    # Green for pixel values 56-110
        (255, 255, 0),  # Yellow for pixel values 111-165
        (255, 100, 0),  # Orange for pixel values 166-220
        (255, 255, 255) # White for pixel values 221-255
    ]
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j][0] 
            if pixel_value <= 55:
                output_image[i, j] = colors[0]
            elif (pixel_value <= 110):
                output_image[i, j] = colors[1]
            elif pixel_value <= 165:
                output_image[i, j] = colors[2]
            elif pixel_value <= 220:
                output_image[i, j] = colors[3]
            else:
                output_image[i, j] = colors[4]

    return output_image

def segmentation_yCbCr(image):
    yCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ThrY = 128
    segmented_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    matrix = np.array([[0.257, 0.505, 0.098],
                    [-0.148, -0.291, 0.439],
                    [0.439, -0.368, -0.071]])
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Y = np.dot(matrix, yCbCr_image[i,j])
            if Y[0]> ThrY:
                segmented_image[i, j] = 255
            else:
                segmented_image[i, j] = 0
           
    return segmented_image
'''
# object labelling
def object_labeling(imgBW):
    tmpheight, tmpwidth= imgBW.shape
    label = np.zeros_like(imgBW)
    rep = 1  
    amount = 1000
    print(imgBW.shape)
    for j in range(rep):
        n = 1
        # Forward scan
        for x in range(1, tmpheight-2):
            for y in range(1, tmpwidth-2):
                if x==1 and y==1 :
                    mask = [
                        label[x, y]
                    ]
                elif x==1 and y!=1:
                    mask = [
                        label[x, y - 1], label[x + 1, y - 1]
                    ]
                elif x!=1 and y==1:
                    mask = [
                        label[x - 1, y], label[x, y]
                    ]
                elif x==tmpwidth-2:
                    mask = [
                        label[x - 1, y - 1], label[x, y - 1],label[x - 1, y], label[x, y]
                    ]
                else:
                    mask = [
                        label[x - 1, y - 1], label[x, y - 1], label[x + 1, y - 1],
                        label[x - 1, y], label[x, y]
                    ]
                if imgBW[x, y] == 1:
                    temp = mask[0] or mask[1] or mask[2] or mask[3]
                    if temp == 0:
                        label[x, y] = n
                        bscan = 1
                        n += 1
                    else:
                        min_val = mask[0]
                        for i in range(1, 5):
                            if min_val == 0:
                                min_val = mask[i]
                                break
                            if mask[i] < min_val and mask[i] != 0:
                                min_val = mask[i]
                        label[x, y] = min_val
                        bscan = 1

        # Backward scan
        for x in range(1, tmpheight - 2):
            for y in range(1, tmpwidth - 2):
                if label[tmpheight - 1 - y, tmpwidth - 1 - x] != 0:
                    mask = [
                        label[tmpwidth - 1 - x - 1, tmpheight - 1 - y - 1],
                        label[tmpwidth - 1 - x, tmpheight - 1 - y - 1],
                        label[tmpwidth - 1 - x + 1, tmpheight - 1 - y - 1],
                        label[tmpwidth - 1 - x - 1, tmpheight - 1 - y],
                        label[tmpwidth - 1 - x, tmpheight - 1 - y]
                    ]
                    min_val = mask[0]
                    for i in range(1, 5):
                        if min_val == 0:
                            min_val = mask[i]
                            break
                        if mask[i] < min_val and mask[i] != 0:
                            min_val = mask[i]
                    label[tmpwidth - 1 - x, tmpheight - 1 - y] = min_val

    # Finish
    count = np.zeros((amount, 5))
    for x in range(tmpheight):
        for y in range(tmpwidth):
            count[label[x, y], 0] += 1
            if count[label[x, y], 1] == 0:
                count[label[x, y], 1] = x
                count[label[x, y], 2] = x
                count[label[x, y], 3] = y
                count[label[x, y], 4] = y
            if x < count[label[x, y], 1] and count[label[x, y], 1] != 0:
                count[label[x, y], 1] = x
            if x > count[label[x, y], 2] and count[label[x, y], 2] != 0:
                count[label[x, y], 2] = x
            if y < count[label[x, y], 3] and count[label[x, y], 3] != 0:
                count[label[x, y], 3] = y
            if y > count[label[x, y], 4] and count[label[x, y], 4] != 0:
                count[label[x, y], 4] = y
    # Assigning random colors to objects
    colors = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)

    # Create a colored image based on labels
    colored_image = np.zeros((tmpheight, tmpwidth, 3), dtype=np.uint8)
    for x in range(tmpheight):
        for y in range(tmpwidth):
            if label[x, y] != 0:
                colored_image[x, y] = count[label[x, y]]
    cv2.imshow('colored image', colored_image)
    return colored_image


labeled_image=object_labeling(segmented_image)
cv2.imshow('Labelled image', labeled_image)
'''

root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("800x500")
root.configure(bg="black")  

info_text = """Brain tumors are abnormal growths of cells in the brain. They can be benign (non-cancerous) or malignant (cancerous). Benign tumors are usually slow-growing and do not spread to other parts of the body, while malignant tumors can grow quickly and spread to nearby tissues. 

Detecting brain tumors typically involves medical imaging techniques such as MRI, CT scan, or PET scan. These imaging tests allow doctors to visualize the brain and identify any abnormal growths. Once a tumor is detected, further diagnostic tests may be performed to determine its type and severity.

Early detection of brain tumors is crucial for effective treatment and improved outcomes. Symptoms of brain tumors can vary depending on their size and location, but may include headaches, seizures, changes in vision or hearing, and cognitive problems.

If you suspect you may have a brain tumor or are experiencing symptoms, it is important to consult a healthcare professional for proper evaluation and diagnosis.

We have designed a meticulous detection system for Brain Tumors which can be vital for doctors and healthcare professionals alike.

The last window will display and pinpoint a tumor (if found) by using a blue region pinpointing.

You can close and exit the image windows by pressing any key"""

info_label = tk.Label(root, text=info_text, font=("Arial", 12), fg="white", bg="black", wraplength=780, justify="center")
info_label.pack(pady=20)

load_button = tk.Button(root, text="Please Select and Load an Image", command=load_image, bg="Black", fg="white")  
load_button.pack(pady=20)

root.mainloop()
