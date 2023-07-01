# Project: Extract information from Vietnamese ID card

#### Create a folder stores raw image(s) to get information from: 
_data/input_

#### Install packages:
> pip install -r requirements.txt  

> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

#### Folder stores raw image(s) to get information from: 
_data/input_

#### Usage: 
From the terminal (in project's directory):  
> python run_this_main.py --input "**_{path to input image}_**"

The output result is created under _result/stage4_ocr_in_csv_ folder.
