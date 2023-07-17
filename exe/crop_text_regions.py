import os
import numpy as np
import cv2


def crop(pts, image):
    """
    :param pts: 4 pairs of coordinates of 4 angles of the text region.
    :param image: extracted-card image.
    :return: Cropped text region image.
    """
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


def generate_words(extracted_img_path, score_bbox, result_path='result/cropped_text_regions'):
    """
    :param extracted_img_path: path to the extracted-card image.
    :param score_bbox: a string of text-region coordinates from CRAFT model execution.
    :param result_path: path to result folder of this function.
    :return: path to result folder of the input image, image name (without extension)
    """
    try:
        image_name = os.path.basename(extracted_img_path)
        image = cv2.imread(extracted_img_path)

        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        result_folder = os.path.join(result_path, image_name_no_ext)

        score_bbox = score_bbox.split('),')
        num_bboxes = len(score_bbox)

        for num in range(num_bboxes):
            bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
            if bbox_coords != ['{}']:
                l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
                t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
                r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
                t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
                r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
                b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
                l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
                b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
                pts = np.array([[int(l_t), int(t_l)], [int(r_t), int(t_r)], [int(r_b), int(b_r)], [int(l_b), int(b_l)]])

                if np.all(pts) > 0:
                    try:
                        word = crop(pts, image)
                        if not os.path.isdir(result_folder):
                            os.makedirs(result_folder)

                        file_name = os.path.join(result_folder,
                                                 image_name_no_ext + '_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.
                                                 format(l_t, t_l, r_t, t_r, r_b, b_r, l_b, b_l))
                        cv2.imwrite(file_name, word)
                        print('Image saved to ' + file_name)

                    except Exception:
                        continue
        if os.path.isdir(result_folder):
            return result_folder, image_name_no_ext
        else:
            print('Cannot crop text regions from {}'.format(extracted_img_path))
            return None, None
    except Exception:
        print('Cannot crop text regions from {}'.format(extracted_img_path))
        return None, None
