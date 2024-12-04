# AmpliVision
AmpliVision is an open source program designed to Automate AMPLI rapid test workflows. Mainly, It is able to load and process AMPLI rapid test images from a specified folder, find the grid and Ampli blocks in the image, read the diagnostic result for each Ampli block, generate a synthetic dataset of images, train a CNN model to classify AMPLI tests with generated images, then run outlier detection. 

<br>

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Pre-Defined Use Cases](#pre-defined-use-cases)
  - [Scan AMPLI test images and extract results](#scan-ampli-test-images-and-extract-results)
  - [Train a CNN LENET model to classify AMPLI test images](#train-a-cnn-lenet-model-to-clasify-ampli-test-images)
  - [Detect Outliers With Trained Model And PyOD](#detect-outliers-with-trained-model-and-pyod)
  - [Test a trained CNN model in Predicting Never Seen AMPLI tests](#test-a-trained-cnn-model-in-predicting-never-seen-ampli-tests)
  - [Visualize Generated Data in Jupyter Notebook](#visualize-generated-data-in-jupyter-notebook)
  - [Display History After Trainings](#display-history-after-trainings)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [Implementing Your AMPLI Tests](#implementing-your-ampli-tests)
- [More Resources](#more-resources)
- [License](#license)
- [Contact](#contact)


<br>

## File Structure
![image](/AmpliVision/_docs/HighLevel.png)

<br>

## Installation
Clone the repository:
```bash
git clone "https://github.com/mattynb/AmpliVision.git"
```

Ensure you have python3.11.9 and the compatible [GPU-Enabled TensorFlow](https://www.tensorflow.org/install/source#gpu) setup in your environment. Then, from the AmpliVision root directory run:
``` python
pip install -r AmpliVision/requirements.txt
```
<br>

## Usage
#### AmpliVision package takes the following arguments:
``` bash
Python AmpliVision USE_CASE DATASET TAG*
```

1. Supported **USE_CASE** parameters:
    -
    CORE
    - [SCAN](#scan-ampli-test-images-and-extract-results)
    - [LENET](#train-a-cnn-lenet-model-to-clasify-ampli-test-images)
    - [PyOD](#detect-outliers-with-trained-model-and-pyod)
    ___
    EXTRA

    - [test](#test-a-trained-cnn-model-in-predicting-never-seen-ampli-tests)
    - [check_data](#visualize-generated-data-in-jupyter-notebook)
    - [history](#display-history-after-trainings)
2. Supported **DATASET** parameters:
    -
    Determines what dataset will be used.
    - [MARKER]() - *The Marker dataset used for proof of concept runs*
    - [YOUR_TARGET](#implementing-your-AMPLI-tests)
    - [_]() - *For use cases that do not need datasets. Such as history.*
3. **TAG***
    -
    optional ID string. Defaults to dataset if None

<br>

## Pre-Defined Use Cases

CORE Use Cases

### Scan AMPLI test images and extract results
Example command to scan and extract MARKER data. The scanned images will be in [data/scanned_MARKER](./AmpliVision/data/scanned_MARKER/) and extracted results in [data/results/DATE](./AmpliVision/data/results/)
``` bash
Python AmpliVision scan MARKER My_Identifiable_TAG
```
### Train a CNN LENET model to clasify Ampli test images
``` python
python3 AmpliVision LENET scanned_MARKER MARKER
```

### Detect Outliers With Trained Model And PyOD
``` python
python3 AmpliVision PyOD scanned_MARKER MARKER
```
---

EXTRA Use Cases

### Test a trained CNN model in Predicting Never Seen Ampli tests
``` python
python3 AmpliVision test scanned_MARKER MARKER
```

### Visualize Generated Data in Jupyter Notebook
``` python
python3 AmpliVision check_data scanned_MARKER MARKER
```

### Display History After Trainings
``` python
python3 AmpliVision history scanned_MARKER MARKER
```


## Contributing
If you'd like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Test your changes thoroughly.
5. Create a pull request.


### Implementing Your AMPLI Tests

1. Add the folder with your AMPLI test images in [data](./AmpliVision/data/). Make sure each file's name start with the result. For example, a test that shows positive result for DENV4 should be named DENV4_...

2. Define the targets of your AMPLI rapid test in [\_\_main\_\_.py](./AmpliVision/__main__.py). "YOUR_TARGET" is the placehoder implementation.

    ``` python
    def manage_targets(dataset):
        """ assigns targets user wants the CNN to predict in specific datasets """
        
        dataset = dataset.upper()

        if "MARKER" in dataset:    
            TAG = 'MARKER'
            TARGETS = ['lung', 'thyroid', 'ovarian', 'prostate', 'skin', 'control', 'breast']
        
        elif "YOUR_TARGET" in dataset:
            # Here is an example to show where you can assign your own targets to your dataset     
            print(" YOUR_TARGET not implemented yet in manage_targets() [__main__.py file], exiting...")
            exit()

    ```

## More Resources
- [Documentation](https://mattynb.github.io/AmpliVision/_docs)
- [Presentation Slides](/AmpliVision/_docs/ABRCMS_oral_presentation.pdf)

## License
This project is licensed under the GNU AGPLv3 - see the [LICENSE](../LICENSE) file for details.

## Contact
If you have any questions or need assistance, feel free to contact us:
- [Matheus Berbet](mailto:matheus.berbet001@umb.edu)
- [Josselyn Mata](mailto:j.matacalidonio001@umb.edu)
- [Daniel Haehn](mailto:daniel.haehn@umb.edu)
- [Kimberly Hamad](mailto:kim.hamad@umb.edu)

We hope you find this repository useful for your automated Ampli needs!
