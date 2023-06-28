import argparse

from exe.detect_and_crop_card import extract_card

parser = argparse.ArgumentParser("Vietnamese ID card extraction")
parser.add_argument("--input", default=None, help="Path to a single raw input image")

args = parser.parse_args()

input_path = args.input

extracted_card_path = extract_card(input_path)
