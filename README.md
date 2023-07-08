# Project: Extract information from Vietnamese ID card

#### Create a folder stores raw image(s) to get information from: 
_data_

#### Install packages:
> pip install -r requirements.txt  

> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

#### Usage: 
From the terminal (in project's directory):  
> python run_this_main.py --input "**_data/...{path to input image}_**"

The output result is created under _result/stage4_ocr_in_csv_ folder.

## Methods and accuracy:
| Stage - Method                                                                                                                | Detailed accuracy                                                                                                                    |
|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Stage 1: Detect ID card from input image.                                                                                     | **Detected 81.53%** <br/>= number of detected card / total input sample.<br/> Note: Depends on the image's quality.                      |
| Stage 2: Apply CRAFT pre-trained model (dataset: ICDAR 2013) to detect text regions (coordinates) from the extracted ID card. | **Precision = 97.4%**<br/>Note: According to [this rank](https://paperswithcode.com/paper/character-region-awareness-for-text-detection). |
| Stage 3: Crop text regions from ID card into separated files.                                                                 | **100%** - just crop images by coordinates.                                                                                              |
| Stage 4: Apply VietOCR model to predict text from cropped text regions.                                                       | **Accuracy = 93%** <br/>Note: calculated by comparing with manual input text, using CER method.                                          |
