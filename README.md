# Project: Extract information from Vietnamese ID card

_Python 3.8_
#### Install packages:
> pip install -r requirements.txt  

> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

#### Folder stores raw image(s) or folder(s) to get information from: 
_data_

#### Usage for a single image: 
From the terminal (in project's directory):  
> python run_this_main.py --input "**_{path to input image}_**"

The output result is in the terminal.

#### Usage for a folder of images: 
From the terminal (in project's directory):  
> python run_this_main.py --folder "**_{path to the folder contains only images}_**"

Input filename for the result file on the screen. The path to output result CSV file is in the terminal.

## Methods and accuracy:
| Stage - Method                                                                                                          | Detailed discrete accuracy                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Stage 1: Apply CRAFT pre-trained model (dataset: ICDAR 2013) to detect text regions (coordinates) from the input image. | **Precision = 97.4%**<br/>Note: According to [this rank](https://paperswithcode.com/paper/character-region-awareness-for-text-detection). |
| Stage 2: Crop text regions from ID card into separated files.                                                           | **100%** - just crop images by coordinates.                                                                                               |
| Stage 3: Apply VietOCR model to predict text from cropped text regions.                                                 | **Accuracy = 93%** <br/>Note: calculated by comparing with manual input text, using CER method.                                           |

### Testing stats: (batch 400 images)
| Measures | Value  |
|----------|--------|
| Precision | 0.9854 |
| Recall   | 0.9883 |
| F1 Score | 0.9868 |
