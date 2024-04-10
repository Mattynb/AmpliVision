# Ampli Recognition System
This program is designed to load and process images from a specified folder, find the grid and Ampli blocks in the image, and then read the diagnostic result for each Ampli block. The bird's-eye view of the system is:

![image](https://github.com/Mattynb/PhaseA_CV_NanoBioLab/assets/93104391/e3d7c205-2f16-4c66-a6ea-9ba37ee79a2e)


## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.12+
- OpenCV (Cv2)
-- ```python pip download cv2```
- Numpy 1.26.4+   
-- ```python pip download numpy```
- Pillow-heif 0.15.0+  
-- ```python pip download pillow-heif```
- Pillow (PIL) 10.2.0+  
-- ```python pip download pillow```

Good to have:
- An IDE for coding such as Visual Studio Code
- Git for version control

### Installation

Clone this repository to your local machine:

```bash
git clone "https://github.com/mattynb/PhaseA_CV_NanoBioLab.git"
```

### Usage
To use this toolkit, follow these steps:

1. Open the main.py script.
2. Set the path_to_imgs variable to the path of the folder containing your images.
3. Run the main function to load and process the images.
``` python
if __name__ == '__main__':
    path_to_imgs = r"C:\Users\YourUser\Desktop\YourImageFolder\*"
    main(path_to_imgs)
```

### Contributing
If you'd like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Test your changes thoroughly.
5.Create a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
If you have any questions or need assistance, feel free to contact me:

Matheus Berbet - matheus.berbet001@umb.edu

We hope you find this repository useful for your image processing needs!
