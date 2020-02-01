# Detecting-Disaster
## Introduction ##

* Disasters have become very prevalent in recent years

* Millions of People are dying and suffering

* Times like these are when timely response is crucial…

* Unfortunately the incidents are not responded to on time; it is always too late

* Severity of the incident is difficult to determine and exact location of the incident is a big challenge itself

![Capturde](https://user-images.githubusercontent.com/55071900/73589713-3afbfd00-4504-11ea-87a6-99f1dbac181c.PNG)

## The Impact of Social Media

* Nowadays everybody has a social media account​
* People tend to post or share any big event in the social network​
* There are many cases when news come in social media (including pictures) faster than it comes through official news channel

#How Social Media will Help us

We want to create a system that:​
Will raise alert and determine degree of damage of any disaster from social media posts​
Inform appropriate authorities based on automated analysis ​
We investigate the potential use of various available deep learning techniques to develop such application ​
Models we are using to run our experiments: VGG 16, VGG 19, Inception-V3, Inception-ResNet-V2

#Schematic View of Proposed  Modification

![image](https://user-images.githubusercontent.com/55071900/73589742-b8277200-4504-11ea-977c-05af024f9b9f.png)

#Dataset

With deep learning comes the requirement of huge and diverse dataset.​
We acquired a dataset from University of California Irvine, (UCI) which contain 5,885 disaster images in total, mostly from Western countries​
We created our own dataset by collecting local disaster images from social media and online news portal. It only consists of 493 images​
We call our dataset South Asian Disaster (SAD) Dataset

#Result of Experiment 01: 5-Fold Cross Validation on UCI Dataset 

![image](https://user-images.githubusercontent.com/55071900/73589875-1b65d400-4506-11ea-9c5e-9d46eaabab25.png)

#Result of Experiment 02: Train with UCI Dataset and Test with SAD Dataset

![image](https://user-images.githubusercontent.com/55071900/73589888-3afcfc80-4506-11ea-91d5-ec72e0106644.png)

#Difference Between UCI and SAD Datasets

![image](https://user-images.githubusercontent.com/55071900/73589767-fd4ba400-4504-11ea-8dfe-5386c94b1b96.png)

#Difference Between Fire Disasters

SAD contains both live fire images and post fire images​
UCI dataset mostly contains live fire images​
Live fire and post fire images are visually very different​
Post fire in fact looks similar to infrastructure damage

![image](https://user-images.githubusercontent.com/55071900/73589914-70a1e580-4506-11ea-8982-775b383bdc91.png)

#Experiment 03: Mixing the UCI dataset and SAD data

Training with UCI dataset and 60% of the SAD dataset ​
Test with the remaining 40% of the SAD dataset. 

![image](https://user-images.githubusercontent.com/55071900/73589935-a34bde00-4506-11ea-9650-ca01355e0e82.png)

#Improvement in Performance Experiment 2 & 3​

8.21% increase in average Precision ​
8.4% increase in average Recall​
2% increase in average Accuracy
