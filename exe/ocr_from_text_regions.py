import os
import pandas as pd
from models.vietocr.utils import init_config, pred_text
from modules.process_output.ocr_processing import sort_ocr_result


def vietocr_all(text_regions_path, img_name_no_ext, result_path='result/stage4_ocr_in_csv'):
    """
    Use VietOCR to read text from text-region image.
    :param text_regions_path: Path to folder contains cropped text-region images.
    :param img_name_no_ext: Original image's name without extension.
    :param result_path: Path to folder contains result file (csv).
    :return: Paths to CSV files (raw result and sorted result) with 2 columns: text-region image file's name and predicted text.
    """
    try:
        # Step 1: Apply VietOCR model to read all cropped text regions and save into a simple dataframe.
        region_files = os.listdir(text_regions_path)

        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        region_file_names = []
        predictions = []

        config = init_config()

        for region in region_files:
            region_file_names.append(region)
            region_path = os.path.join(text_regions_path, region)
            predicted_text = pred_text(region_path, config)
            predictions.append(predicted_text)

        result = pd.DataFrame(columns=['text_region_file', 'predicted_word'])
        result['text_region_file'] = region_file_names
        result['predicted_word'] = predictions
        raw_output_path = os.path.join(result_path, 'RAW_RESULT_' + img_name_no_ext + '.csv')
        result.to_csv(raw_output_path)

        # Step 2: Sort the OCR result to have the top-down order of the result strings.
        sorted_result = sort_ocr_result(result)
        sorted_output_path = os.path.join(result_path, 'SORTED_RESULT_' + img_name_no_ext + '.csv')
        sorted_result.to_csv(sorted_output_path)
        straighten = ' '.join(sorted_result['full_text'])

        return raw_output_path, sorted_output_path, straighten
    except Exception:
        print('Cannot apply VietOCR model for image from {}'.format(text_regions_path))
        return None, None, None
