import pandas as pd


def split_img_name(all_images):
    """
    :param all_images: list of all text-region images' names.
    :return: list of lists that contain all coordinates in a text-region image file.
    """
    split = []
    for i in all_images:
        split.append(i[-0:-4].split("_"))
    return split


def boxes(bbox):
    """
    :param bbox: list of lists that contain all coordinates in a text-region image file.
    :return: list of coordinates pairs (x, y) of all angles of the text region.
    """
    bboxes = []
    for i in range(len(bbox)):
        # transform path to array of coordinator
        arr1 = [int(float(bbox[i][1])), int(float(bbox[i][3])), int(float(bbox[i][5])), int(float(bbox[i][7]))]
        arr2 = [int(float(bbox[i][2])), int(float(bbox[i][4])), int(float(bbox[i][6])), int(float(bbox[i][8]))]
        zipped = zip(arr1, arr2)
        bboxes.append(list(zipped))

    return bboxes


def add_to_data(data, bboxe):
    """
    :param data: (pandas Dataframe) raw ocr result dataframe.
    :param bboxe: list of coordinates pairs (x, y) of all angles of the text region.
    :return: A pandas Dataframe with angle's coordinates columns.
    """
    pt1_x = []
    pt1_y = []
    pt2_x = []
    pt2_y = []
    pt3_x = []
    pt3_y = []
    pt4_x = []
    pt4_y = []

    for i in range(len(data)):
        pt1_x.append(bboxe[i][0][0])
        pt1_y.append(bboxe[i][0][1])
        pt2_x.append(bboxe[i][1][0])
        pt2_y.append(bboxe[i][1][1])
        pt3_x.append(bboxe[i][2][0])
        pt3_y.append(bboxe[i][2][1])
        pt4_x.append(bboxe[i][3][0])
        pt4_y.append(bboxe[i][3][1])

    data.insert(1, 'pt1_x', pt1_x, True)
    data.insert(2, 'pt1_y', pt1_y, True)
    data.insert(3, 'pt2_x', pt2_x, True)
    data.insert(4, 'pt2_y', pt2_y, True)
    data.insert(5, 'pt3_x', pt3_x, True)
    data.insert(6, 'pt3_y', pt3_y, True)
    data.insert(7, 'pt4_y', pt4_x, True)
    data.insert(8, 'pt4_y', pt4_y, True)
    return data


def sorting_detail(new_data):
    """
    :param new_data: A pandas Dataframe with angle's coordinates columns and VietOCR raw result.
    :return: A pandas Dataframe with top-down (according to coordinates) sorted VietOCR result.
    """
    sorted_data = new_data.sort_values(by=['pt1_y'], ascending=True)
    range_pt1_y = [0]

    for i in range(len(sorted_data)):
        try:
            if (sorted_data.iloc[i + 1, 2] - 4) > sorted_data.iloc[i, 2]:
                range_pt1_y.append(i + 1)
        except IndexError:
            range_pt1_y.append(i + 1)
    text = []
    img_name = []
    for i in range(len(range_pt1_y)):
        try:
            text_1 = []
            index1 = range_pt1_y[i]
            index2 = range_pt1_y[i + 1]
            print(f'Checkpoint1: {i}')
            for j in range(len(sorted_data[index1:index2])):
                sort_x = sorted_data[index1:index2].sort_values(by='pt1_x', ascending=True)
                text_1.append(sort_x.iloc[j, 9])
                print(f'Checkpoint2: {j}')
                print(f'Checkpoint3: {sort_x.iloc[j, 9]}')

            text.append(" ".join(text_1))
            img_name.append(sorted_data.iloc[i, 0])
        except IndexError:
            print("Run success.")
    sorted_data = pd.DataFrame(list(zip(img_name, text)),
                               columns=['img_name', 'full_text'])

    return sorted_data


def sort_ocr_result(raw):
    """
    :param raw: (pandas Dataframe) Raw result from VietOCR.
    :return: top-down (according to coordinates) sorted result.
    """
    new_data = add_to_data(raw, bboxe=boxes(split_img_name(raw['text_region_file'])))
    sorted_result = sorting_detail(new_data)

    return sorted_result
