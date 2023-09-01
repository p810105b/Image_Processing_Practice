import numpy as np
import cv2 as cv


def Average_Filter(origin):
    height, width, color = origin.shape
    # padding with the edge value around 
    origin = np.pad(origin, ((1,1), (1,1), (0,0)), 'edge')
    # build a new matrix for processed image
    blurred = np.zeros((height , width , color), np.uint8, 'C')
    for i in range(0, height):
        for j in range(0, width):
            for k in range(color):
                # calculate the average value of the each pixel for 3x3 neighborhood pixels
                for x in range(3):
                    for y in range(3):
                        blurred[i, j, k] += int(origin[i+x, j+y, k]/9 )
    return blurred 
   
   
def Gaussian_kernel_generation(size, sigma): # size = 3, 5, 7, ... 2n+1
    gaussian_kernel = np.zeros((size, size))
    # invalid input size modification
    if size % 2 == 0:
        size = 1 + size
    elif size < 3:
        size = 3
    # fomula for gaussian distribution    
    for i in range(size):
        for j in range(size):
            # middle position
            m = (size - 1) / 2
            x = i - m
            y = j - m
            # index part
            gaussian_kernel[i, j] = -(x**2 + y**2) / (2*(sigma**2))
            
    gaussian_kernel = np.exp(gaussian_kernel) / (2 * np.pi * (sigma**2))
    
    return gaussian_kernel 


def Gaussian_Filter(origin, gaussian_kernel):
    # kernel size
    kernel_size, kernel_size = gaussian_kernel.shape
    pad_size  = int((kernel_size - 1) / 2)
    # for colorful case
    if len(origin.shape) == 3:
        height, width, color = origin.shape
        # padding with the edge value around 
        origin = np.pad(origin, ((pad_size, pad_size), (pad_size, pad_size), (0,0)), 'edge')
        # build a new matrix for processed image
        blurred = np.zeros((height , width , color), np.uint8, 'C')
        # scan each pixel with the kernel 
        for i in range(0, height):
            for j in range(0, width):
                for k in range(color):
                    for x in range(-pad_size, pad_size+1):
                        for y in range(-pad_size, pad_size+1):
                            blurred[i, j, k] += origin[i+pad_size+x, j+pad_size+y, k] * gaussian_kernel[x+pad_size, y+pad_size]                         
    # for grayscale case                     
    else:
        height, width = origin.shape
        # padding with the edge value around  
        origin = np.pad(origin, ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
        # build a new matrix for processed image
        blurred = np.zeros((height , width), np.uint8, 'C')
        # scan each pixels with the kernel     
        for i in range(0, height):
            for j in range(0, width):
                for x in range(-pad_size, pad_size+1):
                        for y in range(-pad_size, pad_size+1):
                            blurred[i, j] += origin[i+pad_size+x, j+pad_size+y] * gaussian_kernel[x+pad_size, y+pad_size]
    return blurred


def Median_Filter(origin):
    height, width, color = origin.shape
    # padding with the edge value around 
    origin = np.pad(origin, ((1,1), (1,1), (0,0)), 'edge')
    # build a new matrix for processed image
    blurred = np.zeros((height , width , color), np.uint8, 'C')
    
    for i in range(0, height):
        for j in range(0, width):
            for k in range(color):
                # list the 3x3 neighborhood pixels
                Neighbor_list = []
                for x in range(3):
                    for y in range(3):
                        Neighbor_list += [origin[i+x, j+y, k]]
                # sort the 3x3 neighbor pixels list
                Sorted_list = sorted(Neighbor_list)
                # choose the 5th-largest(median) value  of the 3x3 neighborhood pixel
                blurred[i, j, k] = Sorted_list[4]       
    return blurred
    
    
def Sobel_edge_detector(origin):
    height, width, color = origin.shape
    # padding with the edge value around 
    origin = np.pad(origin, ((1,1), (1,1), (0,0)), 'edge')
    # build a new matrix for processed image
    result = np.zeros((height , width , color), np.uint8, 'C')
    # define the kernel for x and y direction respectively
    kernel_x = [[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]]
                
    kernel_y = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    
    kernel_x = np.array(kernel_x)
    kernel_y = np.array(kernel_y)
    
    for i in range(0, height):
        for j in range(0, width):
            for k in range(color):
                # scan each 3x3 pixels with the 3x3 kernel
                Gx = 0
                Gy = 0
                for x in range(3):
                    for y in range(3):
                        Gx += origin[i+x, j+y, k] * kernel_x[x, y]
                        Gy += origin[i+x, j+y, k] * kernel_y[x, y]
                
                result[i, j, k] = (Gx**2 + Gy**2)**0.5
                
    return result
    
    
def Laplacian_edge_detector(origin):
    height, width, color = origin.shape
    # padding with the edge value around 
    origin = np.pad(origin, ((1,1), (1,1), (0,0)), 'edge')
    # build a new matrix for processed image
    result = np.zeros((height , width , color), np.uint8, 'C')
    # define the kernel for laplacian filter, which is a better detection than another kernel
    kernel = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    kernel = np.array(kernel)
    
    for i in range(0, height):
        for j in range(0, width):
            for k in range(color):
                # scan each 3x3 pixels with the 3x3 kernel, then sum up           
                for x in range(3):
                    for y in range(3):
                        result[i, j, k] += origin[i+x, j+y, k] * kernel[x, y]
    
    return result
    
   
def Canny_edge_detector(origin, threshold1, threshold2):
    height, width, color = origin.shape
    # Step1：Converse the image to grayscale
    origin = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
    # Reduce noise  
    origin = cv.GaussianBlur(origin, (5, 5), cv.BORDER_DEFAULT)
       
    # Step2：Calculating the gradients
    Gx = cv.Sobel(np.float32(origin), ddepth = -1, dx = 1, dy = 0, ksize = 3)
    Gy = cv.Sobel(np.float32(origin), ddepth = -1, dx = 0, dy = 1, ksize = 3)
      
    # Transfer to polar form
    magnitude, angle = cv.cartToPolar(Gx, Gy, angleInDegrees = True)
    result = magnitude
    # Step3：Non-maximum suppression
    for i in range(width):
        for j in range(height):
            if abs(angle[j, i]) > 180:
                angle[j, i] = abs(angle[j, i] - 180) 
            else: 
                abs(angle[j, i])  
                
            p1_x, p1_y = 0, 0   
            p2_x, p2_y = 0, 0  
            
            # angle = 0
            if 0 < angle[j, i] <= 22.5:
                p1_x, p1_y = i - 1, j
                p2_x, p2_y = i + 1, j  
            # angle = 45
            elif 22.5 < angle[j, i] <= (22.5 + 45):
                p1_x, p1_y = i - 1, j - 1
                p2_x, p2_y = i + 1, j + 1
            # angle = 90
            elif (22.5 + 45) < angle[j, i] <= (22.5 + 90):
                p1_x, p1_y = i, j - 1
                p2_x, p2_y = i, j + 1
            # angle = 135
            elif (22.5 + 90) < angle[j, i] <= (22.5 + 135):
                p1_x, p1_y = i - 1, j + 1
                p2_x, p2_y = i + 1, j - 1
            # angle = 180
            elif (22.5 + 135) < angle[j, i] <= (22.5 + 180):
                p1_x, p1_y = i - 1, j
                p2_x, p2_y = i + 1, j
                
            # suppression
            if width > p1_x >= 0 and height > p1_y >= 0:
                if magnitude[j, i] < magnitude[p1_y, p1_x]:
                    result[j, i] = 0
            if width > p2_x >= 0 and height > p2_y >= 0:
                if magnitude[j, i] < magnitude[p2_y, p2_x]:
                    result[j, i] = 0
                    
    # Step4：Double Thresholding
    magnitude_max = np.max(magnitude)
    # define the upper and lower bound if there are not initial threshold setting 
    if not threshold2 : threshold2 = magnitude_max * 0.8 # strong edge
    if not threshold1 : threshold1 = magnitude_max * 0.2 # weak   edge
                  
    for i in range(width):
        for j in range(height):
            if magnitude[j, i] < threshold1:
                result[j, i] = 0
            
    return result   
      
      
# read image
Path1 = 'Lenna.jpg'
Path2 = 'img.png'

img1  = cv.imread(Path1)
img2  = cv.imread(Path2)  


#instantiate functions
average_1   = Average_Filter  (img1)
average_2   = Average_Filter  (img2)

gauss_kernel = Gaussian_kernel_generation(size=5, sigma=0.8)
Gaussian_1   = Gaussian_Filter(img1, gauss_kernel)
Gaussian_2   = Gaussian_Filter(img2, gauss_kernel)

median_1    = Median_Filter   (img1)
median_2    = Median_Filter   (img2)

sobel_1     = Sobel_edge_detector (img1)  
sobel_2     = Sobel_edge_detector (img2) 

laplacian_1 = Laplacian_edge_detector(img1)
laplacian_2 = Laplacian_edge_detector(img2)

canny_1     = Canny_edge_detector (img1, 60, 80 )            
canny_2     = Canny_edge_detector (img2, 60, 80 )    


# print the output image              
cv.imshow('Origin1_1'  ,img1)
cv.imshow('Origin_2'   ,img2)

cv.imshow('Average_1'  ,average_1)
cv.imshow('Average_2'  ,average_2)

cv.imshow("Gaussian_1", Gaussian_1)
cv.imshow('Gaussian_2' ,Gaussian_2)

cv.imshow('Median_1'   ,median_1)
cv.imshow('Median_2'   ,median_2)

cv.imshow('Sobel_1'    ,sobel_1)
cv.imshow('Sobel_2'    ,sobel_2)

cv.imshow('Laplacian_1',laplacian_1)
cv.imshow('Laplacian_2',laplacian_2)

cv.imshow('Canny_1'    ,canny_1)
cv.imshow('Canny_2'    ,canny_2)


cv.waitKey(0)
cv.destroyAllWindows()                   