# Cloud Forecast Prediction using Deep Learning

This project was put forward due to the work of Keon Roohparvar (@keonroohparvar), Mckenna Reed (@KennaReed), Noah Yuen, and Kevin Sittner (@ksittner), and our work was under the advisement of Dr. Maria Pantoja. 

<!-- SUMMARY OF THE ENTIRE REPO'S FUNCTION -->

The goal of this project is to aid Solar Panel Fields' output efficiency by utilizing Deep Learning techniques to predict how visible the sun is through the clouds 5 minutes into the future. With Solar Panel fields, the energy output is directly reliant on how visible the sun is; if the sun is obscured by any amount of clouds, the energy output of each solar panel will change drasticly in relation to how much sunlight is hitting the solar panel. Because of this, technology that would predict how obscured the sun would be 5 minutes in the future would provide a glimpse of future solar panel efficiency; using this knowledge in tandem with other means of power, like a gas generator, will overall produce a much more stable energy output. 

This technology performs the predictions by 180 degree all-sky-images of the sky, applying segmentation techniques to find out where the clouds are, exporting the cloud's coverage of the past few images to a CSV, and finally utilizing deep learning techniques to predict what the value of the sun's opacity will be in 5 minutes.

## Project Structure

This project is organized in the following sections:
* Data Acquisition
* Image Preprocessing
* Deep Learning Techniques

We will go in detail into each of the portions of the project below.

<!-- Go into each section in detail -->

### Data Acquisition
Data acquisition consisted of using Python's [Selenium](https://selenium-python.readthedocs.io) package to pull images from an [online database](https://www.cameraftp.com/Camera/Cameraplayer.aspx?parentID=229229999&shareID=14125452) of all-sky images at a Solar Panel field located in San Luis Obispo. The python script _pullImages.py_ in the src/ directory is programmed to take a date as an input, and it will pull all images between 10:00AM and 6:00PM from that specified date; it saves them all to a local folder, so this script is used as a tool for acquiring data. 

For our project, we wanted to use our technology as a proof of concept - thus, we chose a date where the sun images were not obscured by any extraordinarry marks. Thus, our code's functionality was on the basis that our image repository was from one day where the sun conditions were fantastic.

### Image Preprocessing
Our image preprocessing consisted of the following steps:

* Finding the Sun
* Dividing the image into directional rings around the sun
* Calculating the percentage of each area covered by clouds utilizing a haze threshold
* Flattening the Images 
* Formatting the data as an input for the neural network models

Each step is explained in the following sections.

#### Finding the Sun in the Image

We decided to take a novel approach to locate the sun, utilizing the image processing library OpenCV to implement a template matching algorithm. It takes a sun template image and our input image as inputs, and output the location of where the sun template matches in the input image. If the sun image is obscured and the template does not match anywhere on the image past a chosen confidence value, we default our sun-detection process to using Python's [Pysolar] (https://pysolar.readthedocs.io/en/latest/) package.

#### Dividing Image into Directional Rings around the Sun

In order to use this image data as input to a neural network, we must convert the pixel data into some form of numerical data that we can input into a model. To do this, we partition the images into concentric rings around the sun and calculate the cloud coverage in each of those rings. We do this by creating four large concentric rings, all sharing their center at the sun's location. After this is done, each of the four rings is split four ways, creating four ring sectors for each of our four initial rings. Thus, this provides 16 points of data that we can feed to our model.

#### Calculating the Percent Covered using a Haze Threshold

To determine the location of the clouds, we employed an algorithm that will extrapolate the clouds by analyzing each pixel's Red, Green, Blue channels. We segment all pixels that pass a certain ratio of their RGB value white, and all pixels that do not pass this ratio become black. Intuitively, this is done to see how "white" a pixel is; if it is extremely white, we classify it as a cloud pixel.

The percent covered metric is calculated by dividing the black pixels by the total number of pixels in each directional ring. For example, if the entire ring is black in the Haze index interpretation, the ring is assumed to have a 100% covered value. This means that the clouds are covering 100% of the image.

#### Flattening the Images
As We employed a stretching algorithm to flatten the sky images, essentially removing the effect of the fish-eye from the camera. Note that these flattened images were only required during the creation of the CNN. 

#### Formatting the Data as an Input for the Neural Network
With the information described above, we converted the input image data to an excel spread sheet. The first 16 columns describe the cloud coverage on each of the 16 sectors as a percentage value out of 100, and the last two columns are the wind direction and speed at that specific time. We use this spread sheet as the input for our machine learning algorithms.

### Deep Learning Techniques
We tested both Long Short-Term Memory and a Convolutional Neural Network architectures in the task of predicting future sun opacity. The LSTM architecture can be found in _src/weatherModel.py_. The CNN architecture can be found in _src/convolutionalModel.py_. After intial testing, we found that the CNN outperformed the LSTM model; we believe it was better able to harness the Wind Direction data.




