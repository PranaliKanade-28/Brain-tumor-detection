# Brain-tumor-detection

Brain hemorrhage is one of second most common causes of deaths. In this project I have tried to analyse brain scan images to identify and highlight brain tumor. I have used mri in dicom format and jpg format for the project.

Part 1: Classification using Python
I have used python for classification of given brain as tumor image or non tumor image. I have used Convolutional Neural Network for modelling. Project uses images in jpg format and converts dicom images to jpg. I have used image augmentation as I had limited brain scan images. Project can process dicom images by converting it to jpg format.

Part 2: Tumor highlight using histogram and k-mean clustering
In this module I am anlysing brain scan image for particular patient. Code is capable of importing dicom images of particular patients file. It can analyse histogram of image and then highlights eroded image using k-mean clustering. Image custering helps identify eroded image i.e. abnormal bright part in image can be segmented as tumor or hemorrhage.
