import argparse

from exe.detect_and_crop_card import extract_card
from exe.craft_text_regions_coordinates import get_text_regions_coordinates
from exe.crop_text_regions import generate_words

parser = argparse.ArgumentParser("Vietnamese ID card extraction")
parser.add_argument("--input", default=None, help="Path to a single raw input image")

args = parser.parse_args()

input_path = args.input

extracted_card_path = extract_card(input_path)
if not extracted_card_path:
    print('Implement CRAFT directly to the input image...')
    extracted_card_path = input_path

bbox_scores_coordinates = get_text_regions_coordinates(extracted_card_path)
text_regions_path, img_name_no_ext = generate_words(extracted_card_path, bbox_scores_coordinates)
