from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


def init_config():
    """
    :return: Customized config for the VietOCR Predictor.
    """
    # config = Cfg.load_config_from_name('vgg_transformer')
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    config['cnn']['pretrained'] = False

    return config


def pred_text(img_path, config):
    """
    Read text from a cropped text region by VietOCR.
    :param config: Customized config for the VietOCR Predictor.
    :param img_path: path to a single text-region image.
    """
    detector = Predictor(config)
    img = Image.open(img_path)
    text = detector.predict(img)

    return text
