import argparse
import os

from exe.craft_text_regions_coordinates import get_text_regions_coordinates
from exe.crop_text_regions import generate_words
from exe.ocr_from_text_regions import vietocr_all

parser = argparse.ArgumentParser("Vietnamese ID card extraction")
parser.add_argument("--input", default=None, help="Path to a single raw input image")

args = parser.parse_args()

input_path = args.input

# remove "_" from image's name
image_name = os.path.basename(input_path)
image_name = image_name.replace('_', '-')
new_input_path = os.path.join(os.path.dirname(input_path), image_name)
os.rename(input_path, new_input_path)
input_path = new_input_path

extracted_card_path = input_path

bbox_scores_coordinates = get_text_regions_coordinates(extracted_card_path)
text_regions_path, img_name_no_ext = generate_words(extracted_card_path, bbox_scores_coordinates)
raw_output_path, sorted_output_path, straighten = vietocr_all(text_regions_path, img_name_no_ext)

print('Result at {}\nStraighten string: {}'.format(sorted_output_path, straighten))
