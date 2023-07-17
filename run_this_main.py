import argparse
import os
import pandas as pd
from exe.craft_text_regions_coordinates import get_text_regions_coordinates
from exe.crop_text_regions import generate_words
from exe.ocr_from_text_regions import vietocr_all
import time

parser = argparse.ArgumentParser("Vietnamese ID card extraction")
parser.add_argument("--input", default=None, help="Path to a single raw input image")
parser.add_argument("--folder", default=None, help="Path to a folder of raw images")

args = parser.parse_args()

input_path = args.input
folder_path = args.folder
start = time.time()

try:
    if not folder_path:
        # remove "_" from image's name
        image_name = os.path.basename(input_path)
        image_name = image_name.replace('_', '-')
        new_input_path = os.path.join(os.path.dirname(input_path), image_name)
        os.rename(input_path, new_input_path)
        input_path = new_input_path

        bbox_scores_coordinates = get_text_regions_coordinates(input_path)
        text_regions_path, img_name_no_ext = generate_words(input_path, bbox_scores_coordinates)
        raw_output_path, sorted_output_path, straighten = vietocr_all(text_regions_path, img_name_no_ext)

        print('Result at {}\nStraighten string: {}'.format(sorted_output_path, straighten))
    else:
        accepted_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]
        im_files = [im_file for im_file in os.listdir(folder_path)
                    if os.path.splitext(im_file)[1].lower() in accepted_formats]
        files_path = []
        strings = []
        valid = []
        for index, im_file in enumerate(im_files):
            input_path = os.path.join(folder_path, im_file)

            # remove "_" from image's name
            image_name = im_file
            image_name = image_name.replace('_', '-')
            new_input_path = os.path.join(folder_path, image_name)
            os.rename(input_path, new_input_path)
            input_path = new_input_path

            if input_path:
                bbox_scores_coordinates = get_text_regions_coordinates(input_path)
                if bbox_scores_coordinates:
                    text_regions_path, img_name_no_ext = generate_words(input_path, bbox_scores_coordinates)
                    if text_regions_path and img_name_no_ext:
                        raw_output_path, sorted_output_path, straighten = vietocr_all(text_regions_path, img_name_no_ext)
                        if raw_output_path and sorted_output_path and straighten:
                            files_path.append(input_path)
                            strings.append(straighten)
                            print('Straighten string: {}\n'.format(straighten))
                            straighten = straighten.lower()
                            is_valid = "chứng minh nhân dân" in straighten or "căn cước công dân" in straighten
                            valid.append(is_valid)

                            print('{}/{}: Result at {}\n'.
                                  format(index+1, len(im_files), sorted_output_path))
                        else:
                            files_path.append(input_path)
                            strings.append('')
                            is_valid = False
                            valid.append(is_valid)
                    else:
                        files_path.append(input_path)
                        strings.append('')
                        is_valid = False
                        valid.append(is_valid)
                else:
                    files_path.append(input_path)
                    strings.append('')
                    is_valid = False
                    valid.append(is_valid)
            else:
                files_path.append(input_path)
                strings.append('')
                is_valid = False
                valid.append(is_valid)

            result_df = pd.DataFrame(list(zip(files_path, strings, valid)),
                                     columns=['files_path', 'strings', 'is_valid'])
            result_path = 'result/check_valid'
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            file_name = 'result_{}_over_{}.csv'.format(index, len(im_files))
            result_path = os.path.join(result_path, file_name)
            result_df.to_csv(result_path)
except Exception as e:
    print(e)

print('PROCESS COMPLETED in {} seconds'.format(time.time() - start))
