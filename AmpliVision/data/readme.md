# Data Folder
Inside this folder there are a few subfolders essential to the user. In this ReadMe I am to explain what the purpose of each is in chronological order. 
-------------

### MARKER, NP, ...
These are the original images taken with a cellphone camera of physical AMPLI tests.

### scanned_MARKER, scanned_NP, ...
These are the scanned images, after initial processing that normalizes and isolates the Grid in images.

### results
Here you can expect to find the **extracted RGB results** of each test spot in the scanned images. Subfolders represent the date which they were scanned. Inside the CSV files containing the extracted results of each image, you will find information about the test such as: date, time, grid_index, block_type, Test Spot1 R,G,B, Test Spot 2 R,G,B, Background Spot R,G,B, and the corrected RGBs (Background spot RGB - Spot X RGB).

### test_components
This folder contains the no background test components used to generate synthetic images that we create to train the ML models later. **If you need tests with new AMPLI blocks, make sure to add them here**

### generated_images 
By default, the generated images are not saved every time a model is trained. However, this folder includes a few examples of what a generated image looks like. The major different is the noise. Inside blank subfolder you will find what a generated image looks like before "painting" the result spots and adding noise.

### ML_perform
This folder will have the [pickle dump](https://docs.python.org/3/library/pickle.html) of the ML training history and the Loss/epoch Accuracy/epoch graphs, all of which will be updated at each epoch of training.

### ML_models
After training is done for each epoch, this folder will have saved the trained ML model.