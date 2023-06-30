import cv2
import os
from modules.process_input import image_processing
from modules.process_input import corners_algorithm


def extract_card(img_path, result_path='result/stage1_extract_card', new_height=500.0):
    """
    :param img_path: path to input (raw) image
    :param result_path: (optional) path to result folder to save the output image
    :param new_height: (optional, default new_height=500.0) expected new height of the resized image
    :return: output image path of extracted card from the raw input
    """
    try:
        image = cv2.imread(img_path)
        image_name = os.path.basename(img_path)

        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        # rescale the image
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        orig_image = image.copy()

        ratio = orig_height / new_height
        new_width = orig_width / ratio

        rescaled_image = cv2.resize(image, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)

        # pre-process the image
        constants = image_processing.get_constants(height=new_height, width=new_width)

        gray_image = image_processing.to_gray(rescaled_image)

        dilated_image = image_processing.to_dilated(gray_image, constants.get('MORPH'))

        edged_image = image_processing.to_edged(dilated_image, constants.get('CANNY'))

        # find 4 most potential corners of the ID card in the RESCALED image (from algorithm)
        (most_potential_4_corners, is_accept) = corners_algorithm.get_corners(edged_image)

        # is_accept = True -> an ID card is detected and cropped, False -> pass and come to the next image
        if not is_accept:
            print('Cannot detect a card from {}'.format(img_path))
            return None

        # 4 most potential corners of the ID card in the ORIGINAL image
        corners_orig = most_potential_4_corners * ratio
        # sort points in clockwise order from the top-left
        corners_orig = corners_algorithm.sort_points(corners_orig)

        result = image_processing.to_bird_eye_view(orig_image, corners_orig)

        output_path = os.path.join(result_path, image_name)

        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        return output_path
    except Exception:
        return None


# path_input_data = {
#     'cccd_chip': '../data/input/cccd_chip',
#     'cccd_nochip': '../data/input/cccd_nochip',
#     'cmnd_cu': '../data/input/cmnd_cu',
#     'cmnd_moi': '../data/input/cmnd_moi'
# }
#
# path_result = '../data/card_cropped'
#
# accepted_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]
#
# for key in path_input_data.keys():
#     input_folder_path = path_input_data[key]
#
#     im_files = [im_file for im_file in os.listdir(input_folder_path)
#                 if os.path.splitext(im_file)[1].lower() in accepted_formats]
#
#     for im_file in im_files:
        # image processing
