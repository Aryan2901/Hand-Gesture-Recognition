# Hand-Gesture-Recognition
I have trained a model that is classifying the hand gesture captured by the camera into the ASL(American Sign Language).

### Constraint

### Main Working:

1. Image  is taken via inbuilt webcam of the system.
2. We see a frame around the hand 
3. Image is converted from BGR to HSV
4. Hand is extracted from the image
5. The gesture is passed to the model
6. Model predicts the hand gesture and prints the predicted letter along with accuracy percentage.

### Dataset used 
Dataset used here was taken from the kaggle. You can access the dataset from the [link](https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset). 

## Working
You have to adjust the converter and other values accordingly.

![image](https://user-images.githubusercontent.com/77443958/236634392-69d34530-0216-4aa3-9ca5-5b5730468127.png) ![image](https://user-images.githubusercontent.com/77443958/236634463-1bd3c388-2aa5-4d03-b39d-141b9da1cc7b.png)

### Confusion Matrix for prediction on **Real Hand Gestures taken from the Camera**

![image](https://user-images.githubusercontent.com/77443958/236634599-19637ca6-e2f6-4299-b510-69192cf836f1.png)
