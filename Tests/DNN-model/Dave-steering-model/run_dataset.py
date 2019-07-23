import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import driving_data
import time
import TensorFI as ti

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

resFile = open("eachFIres.csv", "a")
eachFI = open("eachInjtionRes.csv", "a")

fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)


imgIndex = 140
#while(cv2.waitKey(10) != ord('q')):


for i in range(1):  # num of imgs to be injected
    full_image = scipy.misc.imread("driving_dataset/" + str(imgIndex) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    '''    
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
#    call("clear")
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])
    resFile.write(`i` + "," + `degrees` + "," + `driving_data.ys[i]` + "\n")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1 

    '''

    fi.turnOffInjections()
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
    golden = degrees
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])



    fi.turnOnInjections() 

    totalFI = 0.
    sdcCount = 0 

    fiCount = 100
    for k in range(fiCount): 
        # steering angle under fault
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 

        totalFI += 1
        if(abs(degrees - golden) < 5):
            print(fiCnt, "no SDC")
            resFile.write(`1` + ",")    
            sdc = 1   
        else:
            print(fiCnt, "SDC")
            sdcCount += 1
            resFile.write(`0` + ",")
            sdc = 0
        eachFI.write( `degrees` + ","  + `sdc` + "," + `golden` + "," + `ti.faultTypes.indexOfInjectedData` + "," + `ti.faultTypes.indexOfInjectedBit` + "\n" )
     
        print(totalFI , " Predicted steering angle: " + str(degrees) + " golden", golden)


    resFile.write("\n")    
    print("sdc", sdcCount/totalFI , totalFI)

        
#        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
#        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
#        #and the predicted angle
#        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
#        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
#        dst = cv2.warpAffine(img,M,(cols,rows))
#        cv2.imshow("steering wheel", dst)
        
    
cv2.destroyAllWindows()
