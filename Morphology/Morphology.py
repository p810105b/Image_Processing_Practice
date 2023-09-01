import cv2 as cv
import numpy as np
  

def Transfer_binary(img):
    height, width, color = img.shape
    img_binary = np.zeros((height , width), dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(color):
                if img[i, j, k] > 63:
                    img_binary[i, j] = 255
                else:
                    img_binary[i, j] = 0
    return img_binary


def Dilation(img):
    height, width = img.shape
    img_dilated = np.zeros((height, width), dtype = np.uint8)
    # padding with the edge value around 
    img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
    for i in range(height):
        for j in range(width):
            # 3 * 3 array multiplication 
            product = img[i : i+s_size, j : j+s_size] * structuring_element
            # only 0 and 255 for pixel intensity values
            if np.max(product) > 127:
                img_dilated[i, j] = 255
            else:
                img_dilated[i, j] = 0
    return img_dilated
  
    
def Erosion(img):
    height, width = img.shape
    img_eroded  = np.zeros((height, width), dtype = np.uint8)
    # padding with the edge value around 
    img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
    for i in range(height):
        for j in range(width):
            # 3 * 3 array multiplication 
            product = img[i : i+s_size, j : j+s_size] * structuring_element
            # only 0 and 255 for pixel intensity values
            if np.min(product) > 127:
                img_eroded[i, j] = 255
            else:
                img_eroded[i, j] = 0
    return img_eroded


def Opening(img):    
    height, width = img.shape
    Opening_img = np.zeros((height, width), dtype = np.uint8)
    Eroded_temp = np.zeros((height, width), dtype = np.uint8)
    # First : Erosion , next : Dilation
    Eroded_temp = Erosion(img)
    Opening_img = Dilation(Eroded_temp)
    return Opening_img
    

def Closing(img): 
    height, width = img.shape
    Closing_img = np.zeros((height, width), dtype = np.uint8)
    Dilated_temp = np.zeros((height, width), dtype = np.uint8)
    # First : Dilation , next : Erosion
    Dilated_temp = Dilation(img)
    Closing_img  = Erosion(Dilated_temp)    
    return Closing_img
  
    
def Edge_detector_type1(img):
    height, width = img.shape
    edge_img     = np.zeros((height, width), dtype = np.uint8)
    Dilated_temp = np.zeros((height, width), dtype = np.uint8)
    Closing_temp = np.zeros((height, width), dtype = np.uint8)
    
    Dilated_temp = Dilation(img)
    Closing_temp = Closing (img)
    
    edge_img = Dilated_temp - Closing_temp
    return edge_img
    

def Edge_detector_type2(img):
    height, width = img.shape
    edge_img     = np.zeros((height, width), dtype = np.uint8)
    Eroded_temp  = np.zeros((height, width), dtype = np.uint8)
    Opening_temp = np.zeros((height, width), dtype = np.uint8)
    
    Eroded_temp  = Erosion (img)
    Opening_temp = Opening (img)
    
    edge_img = Opening_temp - Eroded_temp
    return edge_img
    
    
def Laplacian_edge_detector(img):
    height, width= img.shape
    # padding with the edge value around 
    img = np.pad(img, ((1,1), (1,1)), 'edge')
    # build a new matrix for processed image
    edge_img = np.zeros((height , width), np.uint8, 'C')
    # define the kernel for laplacian filter, which is a better detection than another kernel
    kernel = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    kernel = np.array(kernel)
    
    for i in range(0, height):
        for j in range(0, width):
            # scan each 3x3 pixels with the 3x3 kernel, then sum up           
            for x in range(3):
                for y in range(3):
                    edge_img[i, j] += img[i+x, j+y] * kernel[x, y]
    return edge_img


def Sobel_edge_detector(img):
    height, width = img.shape
    # padding with the edge value around 
    img = np.pad(img, ((1,1), (1,1)), 'edge')
    # build a new matrix for processed image
    edge_img = np.zeros((height , width), np.uint8, 'C')
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
            # scan each 3x3 pixels with the 3x3 kernel
            Gx = 0
            Gy = 0
            for x in range(3):
                for y in range(3):
                    Gx += img[i+x, j+y] * kernel_x[x, y]
                    Gy += img[i+x, j+y] * kernel_y[x, y]
            # transfer to binary image
            if (Gx**2 + Gy**2)**0.5 > 15:    
                edge_img[i, j] = 255
            else:
                edge_img[i, j] = 0
    return edge_img

    
def Canny_edge_detector(img, threshold1, threshold2):
    # Step1：Converse the image to grayscale
    if len(img.shape) == 3:
        height, width, color = img.shape
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        height, width = img.shape
    
    # Reduce noise  
    img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
       
    # Step2：Calculating the gradients
    Gx = cv.Sobel(np.float32(img), ddepth = -1, dx = 1, dy = 0, ksize = 3)
    Gy = cv.Sobel(np.float32(img), ddepth = -1, dx = 0, dy = 1, ksize = 3)
      
    # Transfer to polar form
    magnitude, angle = cv.cartToPolar(Gx, Gy, angleInDegrees = True)
    edge_img = magnitude
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
                    edge_img[j, i] = 0
            if width > p2_x >= 0 and height > p2_y >= 0:
                if magnitude[j, i] < magnitude[p2_y, p2_x]:
                    edge_img[j, i] = 0
                    
    # Step4：Double Thresholding
    magnitude_max = np.max(magnitude)
    # define the upper and lower bound if there are not initial threshold setting 
    if not threshold2 : threshold2 = magnitude_max * 0.8 # strong edge
    if not threshold1 : threshold1 = magnitude_max * 0.2 # weak   edge
                  
    for i in range(width):
        for j in range(height):
            if magnitude[j, i] < threshold1:
                edge_img[j, i] = 0
    # transfer to binary img 
    for i in range(height):
        for j in range(width):
            if edge_img[i, j] > 127:
                edge_img[i, j] = 255
            else:
                edge_img[i, j] = 0
    return edge_img   

    
PATH1 = 'morphology.png'
PATH2 = 'morphology_1.png'

img1 = cv.imread(PATH1)
img2 = cv.imread(PATH2) 


# Define the structuring element
# k= 3 for 3*3 sizes of the structuring element
s_size = 3
structuring_element = np.ones((s_size, s_size), dtype = np.uint8)
pad_size = (s_size - 1) // 2


# function instantiation
img1_binary  = Transfer_binary(img1)
img2_binary  = Transfer_binary(img2)

img1_dilated = Dilation(img1_binary)
img2_dilated = Dilation(img2_binary)
img1_eroded  = Erosion (img1_binary)
img2_eroded  = Erosion (img2_binary)

img1_opening = Opening (img1_binary)
img2_opening = Opening (img2_binary)
img1_closing = Closing (img1_binary)
img2_closing = Closing (img2_binary)

img1_edge_type1 = Edge_detector_type1(img1_binary)
img1_edge_type2 = Edge_detector_type2(img1_binary)
img2_edge_type1 = Edge_detector_type1(img2_binary)
img2_edge_type2 = Edge_detector_type2(img2_binary)

laplacian_img1 = Laplacian_edge_detector(img1_closing)
laplacian_img2 = Laplacian_edge_detector(img2_opening)
sobel_img1     = Sobel_edge_detector    (img1_closing)  
sobel_img2     = Sobel_edge_detector    (img2_opening) 
canny_img1     = Canny_edge_detector    (img1_closing, 40, 160 )            
canny_img2     = Canny_edge_detector    (img2_opening, 40, 160 )  

cv.imshow('orign_img1'  , img1)
cv.imshow('orign_img2'  , img2)
cv.imshow('binary_img1' , img1_binary)
cv.imshow('binary_img2' , img2_binary)

cv.imshow('dilated_img1', img1_dilated)
cv.imshow('dilated_img2', img2_dilated)
cv.imshow('eroded_img1' , img1_eroded)
cv.imshow('eroded_img2' , img2_eroded)

cv.imshow('Opening_img1', img1_opening)
cv.imshow('Opening_img2', img2_opening)
cv.imshow('Closing_img1', img1_closing)
cv.imshow('Closing_img2', img2_closing)

cv.imshow('img1_edge_type1', img1_edge_type1)
cv.imshow('img1_edge_type2', img1_edge_type2)
cv.imshow('img2_edge_type1', img2_edge_type1)
cv.imshow('img2_edge_type2', img2_edge_type2)

cv.imshow('Laplacian_img1' , laplacian_img1)
cv.imshow('Laplacian_img2' , laplacian_img2)
cv.imshow('sobel_img1'     , sobel_img1)
cv.imshow('sobel_img2'     , sobel_img2)
cv.imshow('canny_img1'     , canny_img1)
cv.imshow('canny_img2'     , canny_img2)

cv.waitKey(0)
cv.destroyAllWindows()