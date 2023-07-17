import argparse
import os
import pandas as pd
from exe.craft_text_regions_coordinates import get_text_regions_coordinates
from exe.crop_text_regions import generate_words
from exe.ocr_from_text_regions import vietocr_all
from exe.ocr_from_text_regions import check_valid_predicted
import time


def remove_underscore(path_to_img):
    """
    Replace the underscore "_" by a dash "-" in the input image name.
    :param path_to_img: Path to the input image.
    :return: Changed path to the input image after changing the name of the original input image directly.
    """
    image_name = os.path.basename(path_to_img)
    image_name = image_name.replace('_', '-')
    new_input_path = os.path.join(os.path.dirname(input_path), image_name)
    os.rename(input_path, new_input_path)
    return new_input_path


parser = argparse.ArgumentParser("Vietnamese ID card extraction")
parser.add_argument("--input", default=None, help="Path to a single raw input image")
parser.add_argument("--folder", default=None, help="Path to a folder of raw images")

args = parser.parse_args()

input_path = args.input
folder_path = args.folder
start = time.time()

try:
    # Input is a single image
    if not folder_path:
        input_path = remove_underscore(input_path)

        bbox_scores_coordinates = get_text_regions_coordinates(input_path)
        text_regions_path, img_name_no_ext = generate_words(input_path, bbox_scores_coordinates)
        raw_output_path, sorted_output_path, straighten = vietocr_all(text_regions_path, img_name_no_ext)
        is_valid = check_valid_predicted(straighten)

        print('Result at {}\nStraighten string: {}\nValid/Invalid? {}'.format(sorted_output_path, straighten, is_valid))

    # Input is a path to a folder
    else:
        accepted_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]
        im_files = [im_file for im_file in os.listdir(folder_path)
                    if os.path.splitext(im_file)[1].lower() in accepted_formats]
        files_path = []
        strings = []
        valid = []
        for index, im_file in enumerate(im_files):
            input_path = os.path.join(folder_path, im_file)
            input_path = remove_underscore(input_path)

            bbox_scores_coordinates = get_text_regions_coordinates(input_path)
            text_regions_path, img_name_no_ext = generate_words(input_path, bbox_scores_coordinates)
            raw_output_path, sorted_output_path, straighten = vietocr_all(text_regions_path, img_name_no_ext)

            files_path.append(input_path)
            if not straighten:
                straighten = ''

            strings.append(straighten)
            print('Straighten string: {}\n'.format(straighten))
            is_valid = check_valid_predicted(straighten)
            valid.append(is_valid)
            print('{}/{}: Result at {}\nValid/Invalid? {}'.
                  format(index + 1, len(im_files), sorted_output_path, is_valid))

            result_df = pd.DataFrame(list(zip(files_path, strings, valid)),
                                     columns=['files_path', 'strings', 'is_valid'])

            result_path = 'result/check_valid'
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            file_name = 'results_{}_over_{}.csv'.format(index + 1, len(im_files))
            result_path = os.path.join(result_path, file_name)
            result_df.to_csv(result_path)
except Exception as e:
    print(e)

print('PROCESS COMPLETED in {} seconds'.format(time.time() - start))
