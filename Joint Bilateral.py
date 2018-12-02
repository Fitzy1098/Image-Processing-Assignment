import cv2
import numpy

def gauss(x, sigma):
    gaussianFunction=(1.0/(((2*numpy.pi)**0.5)*sigma))*numpy.exp(-(x**2)/(2*(sigma**2)))      #Applies the Gaussian function
    return gaussianFunction

def distance(x1, y1, x2, y2):
    distance=numpy.sqrt(numpy.abs(((x1-x2)**2)+((y1-y2)**2)))     #Calculates the distance between 2 pixels
    return distance

def distance2(diameter,sigma):          #Creates a list of gaussian function values for pixels in the neighbourhood
    middle=(diameter+1)/2               #Finds co-ordinates for centre of neighbourhood 
    middle-=1
    gausses=[]
    for i in range(diameter):           #Iterates through all pixel co-ordinates in neighbourhood
        for j in range(diameter):
            dist=distance(i, j, middle, middle)         #Calculates distance between centre and current co-ordinates
            g=gauss(dist, sigma)                        #Calculates the gaussian function 
            gausses.append(g)               #Appends the value to the list    
    return gausses
           
    
    
def joint_bilateral_filter(image, image1, output, diameter, intensity_sigma, spatial_sigma):  #Applies the joint bilaeral filter to the 2 images    
    radius=diameter/2
    numRows=len(image)
    numColumns=len(image[0])
    gausses=distance2(diameter, spatial_sigma)
    for row in range(numRows):       #Iterates through the rows of image
        for column in range(numColumns):    #Iterates through the columns of image
            for z in range(0,3):                #Iterates through the 3 colour channels
                ks=0                            #Normalisation factor
                filtered=0                      #Output for pixel channel
                point=0
                for i in range(diameter):       #Iterates through the neighbourhood 
                    for j in range(diameter):
                        near_x=int(row-(radius-i))      #Finds x co-ordinate of a pixel in the neighbourhood
                        near_y=int(column-(radius-j))   #Finds x co-ordinate of a pixel in the neighbourhood
                        if near_x>=numRows:              #Adjusts the x co-ordinate if it is outside the range of the image
                            near_x-=numRows
                        if near_y>=numColumns:           #Adjuts the y co-ordinate if it is outside the range of the image
                            near_y-=numColumns
                        gauss_spatial=gausses[point]
                        gauss_intensity = gauss(int(image1[near_x][near_y][z])-int(image1[row][column][z]), intensity_sigma) #Calculate the Gaussian function on the difference in intensities of the two pixels
                        #gauss_spatial = gauss(distance(near_x, near_y, row, column), spatial_sigma)                 #Calcualtes the Gaussian function on the distance between the pixels
                        k=gauss_spatial*gauss_intensity         #Multipliy the two functions togther
                        filtered+=(image[near_x][near_y][z]*k)      #Sum together the functions multiplied by the intensity value for the channel of the pixel in the original image
                        ks+=k                                   #Sum together the functions
                        point+=1
                filtered=filtered//ks                         #Divide by the normalisation factor and round to an integer
                output[row][column][z]=filtered  #Set the pixel channel value to the new value rounded to the nearest whole number
    return output  #Return the filtered image

def run():
    windowName="Smoothed Image"     #Names the display window

    image=cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR)       #Reads the ambient image
    image1=cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR)      #Reads the flash image

    if not image is None and not image1 is None:        #Checks if both files are images
        output=image.copy()                             #Creates a copy of the input image to append the values to
        img_own = joint_bilateral_filter(image, image1, output, 13, 2, 3) #ambient image, flash image, the output image, the diameter of the neighbourhood, intensity sigma, spatial sigma
        cv2.imwrite("filtered_image.jpg", img_own)       #Writes the image to a file

        cv2.imshow(windowName, img_own)     #Displays the filtered image in a window
        key = cv2.waitKey(0)
        if (key == ord('x')):               #Closes the window if the 'x' key is pressed
            cv2.destroyAllWindows()
    else:
        print("An image file did not successfully load.")     #Prints this statement if one or both images fail to load
    
run() #Runs the program
    
