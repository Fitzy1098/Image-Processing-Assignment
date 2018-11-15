import cv2
import numpy

def gauss(x,sigma):
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))      #Applies the Gaussian function

def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))     #Calculates the distance between 2 pixels

def joint_bilateral_filter(image, image1, output, diameter, intensity_sigma, spatial_sigma):  #Applies the joint bilaeral filter to the 2 images    

    for row in range(len(image)):       #Iterates through the rows of image
        for column in range(len(image[0])):    #Iterates through the columns of image
            for z in range(0,2):                #Iterates through the 3 colour channels
                ks=0                            #Normalisation factor
                filtered=0                      #Output for pixel channel
                for i in range(diameter):       #Iterates through the neighbourhood 
                    for j in range(diameter):
                        near_x=int(row-(diameter/2-i))      #Finds x co-ordinate of a pixel in the neighbourhood
                        near_y=int(column-(diameter/2-j))   #Finds x co-ordinate of a pixel in the neighbourhood
                        if near_x>=len(image):              #Adjusts the x co-ordinate if it is outside the range of the image
                            near_x-=len(image)
                        if near_y>=len(image[0]):           #Adjuts the y co-ordinate if it is outside the range of the image
                            near_y-=len(image[0])

                        gauss_intensity = gauss(image1[near_x][near_y][z]-image1[row][column][z], intensity_sigma) #Calculate the Gaussian function on the difference in intensities of the two pixels
                        gauss_spatial = gauss(distance(near_x, near_y, row, column), spatial_sigma)                 #Calcualtes the Gaussian function on the distance between the pixels
                        k=gauss_spatial*gauss_intensity         #Multipliy the two functions togther
                        filtered+=(image[near_x][near_y][z]*k)      #Sum together the functions multiplied by the intensity value for the channel of the pixel in the original image
                        ks+=k                                   #Sum together the functions
                filtered=filtered // ks                         #Divide by the normalisation factor and round to an integer
                output[row][column][z]=filtered  #Set the pixel channel value to the new value rounded to the nearest whole number
    return output

def run():
    windowName="Smoothed Image"     #Names the display window

    image=cv2.imread('./test3a.jpg',cv2.IMREAD_COLOR)       #Reads the ambient image
    image1=cv2.imread('./test3b.jpg',cv2.IMREAD_COLOR)      #Reads the flash image

    if not image is None and not image1 is None:
        #filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
        #cv2.imwrite("image.png",image)
        output=image.copy()
        #img_own = bilateral_filter(image, image1, output, 15, 35, 35)
        #cv2.imwrite("filtered_1.jpg", img_own)
        #output=image.copy()
        
        #img_own = bilateral_filter(image, image1, output, 7, 10, 10)
        #cv2.imwrite("filtered_3.jpg", img_own)
        #output=image.copy()

        #img_own = bilateral_filter(image, image1, output, 20, 20, 20)
        #cv2.imwrite("filtered_4.jpg", img_own)
        #output=image.copy()
        
        #img_own = bilateral_filter(image, image1, output, 7, 20, 20) #ambient image, flash image, the output image, the diameter of the neighbourhood, intensity sigma, spatial sigma
        #cv2.imwrite("filtered_test2.jpg", img_own)

        #img_own = joint_bilateral_filter(image, image1, output, 7, 10, 30) #ambient image, flash image, the output image, the diameter of the neighbourhood, intensity sigma, spatial sigma
        #cv2.imwrite("filtered1.jpg", img_own)
        
        img_own = joint_bilateral_filter(image, image1, output, 15, 10, 40) #ambient image, flash image, the output image, the diameter of the neighbourhood, intensity sigma, spatial sigma
        cv2.imwrite("filtered2.jpg", img_own)
        
        cv2.imshow(windowName, img_own)
        key = cv2.waitKey(0)
        if (key == ord('x')):
            cv2.destroyAllWindows()
    else:
        print("No image file successfully loaded.")
    
    

def hue(pixel):
    hue=0
    RGB=[]
    RGB[0]=pixel[0]/255 #B
    RGB[1]=pixel[1]/255 #G
    RGB[2]=pixel[2]/255 #R
    if RGB[2]>=RGB[1] and RGB[2]>=RGB[0]:
        hue=(RGB[1]-RGB[0])/(max(RGB)-min(RGB))
    if RGB[1]>=RGB[2] and RGB[1]>=RGB[0]:
        hue=2.0+(RGB[0]-RGB[2])/(max(RGB)-min(RGB))
    if RGB[0]>=RGB[2] and RGB[0]>=RGB[1]:
        hue=4.0+(RGB[2]-RGB[1])/(max(RGB)-min(RGB))
    hue*=60
    if hue<0:
        hue+=360
    return hue
    
