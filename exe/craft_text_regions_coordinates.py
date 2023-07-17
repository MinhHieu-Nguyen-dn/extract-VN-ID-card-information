import urllib.request
import os
import time
import torch
import torch.backends.cudnn as cudnn
import cv2
from models.craft import test, imgproc, file_utils
from models.craft.craft import CRAFT
from models.craft.refinenet import RefineNet


def craft_constants():
    """
    :return: All constants for using CRAFT model.
    """
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    cuda = False
    canvas_size = 1280
    mag_ratio = 1.5
    poly = False
    show_time = False
    refine = True
    trained_model = "craft_mlt_25k.pth"
    pretrained_weight_url = "https://drive.google.com/uc?authuser=0&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download"
    refiner_model = "craft_refiner_CTW1500.pth"
    refiner_weight_url = "https://drive.google.com/u/0/uc?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO&export=download"

    return (text_threshold, low_text, link_threshold,
            cuda, canvas_size, mag_ratio, poly,
            show_time, refine, trained_model, pretrained_weight_url, refiner_model, refiner_weight_url)


def get_text_regions_coordinates(image_path, result_path='result/craft_text_regions'):
    """
    :param image_path: path to extracted-card image.
    :param result_path: path to result folder of this function.
    :return: A string contains all bounding box coordinates for text regions in the extracted-card.
    """
    try:
        image_name = os.path.basename(image_path)

        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        # get constants
        (text_threshold, low_text, link_threshold,
         cuda, canvas_size, mag_ratio, poly,
         show_time, refine, trained_model, pretrained_weight_url, refiner_model, refiner_weight_url) = craft_constants()

        # load net and initialize
        net = CRAFT()

        weights_path = 'weights/craft'

        if not os.path.isdir(weights_path):
            os.makedirs(weights_path)

        # Load pre-trained weight file
        pretrained_model_path = os.path.join(weights_path, trained_model)
        if not os.path.exists(pretrained_model_path):
            print('Downloading pre-trained weight file...')
            urllib.request.urlretrieve(pretrained_weight_url, pretrained_model_path)
        else:
            print('Pre-trained model existed at {}'.format(pretrained_model_path))

        print('Loading weights from checkpoint (' + trained_model + ')')
        # if cuda:
        #     net.load_state_dict(test.copyStateDict(torch.load(pretrained_model_path)))
        #     net = net.cuda()
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = False
        # else:
        #     net.load_state_dict(test.copyStateDict(torch.load(pretrained_model_path, map_location='cpu')))

        net.load_state_dict(test.copyStateDict(torch.load(pretrained_model_path, map_location='cpu')))

        net.eval()

        # Load refiner model weight file

        refiner_model_path = os.path.join(weights_path, refiner_model)
        if not os.path.exists(refiner_model_path):
            print('Downloading refiner model file...')
            urllib.request.urlretrieve(refiner_weight_url, refiner_model_path)
        else:
            print('Refiner model existed at {}'.format(refiner_model_path))

        # Link Refiner
        refine_net = None
        if refine:
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
            # if cuda:
            #     refine_net.load_state_dict(test.copyStateDict(torch.load(refiner_model_path)))
            #     refine_net = refine_net.cuda()
            #     refine_net = torch.nn.DataParallel(refine_net)
            # else:
            #     refine_net.load_state_dict(test.copyStateDict(torch.load(refiner_model_path, map_location='cpu')))

            refine_net.load_state_dict(test.copyStateDict(torch.load(refiner_model_path, map_location='cpu')))

            refine_net.eval()
            poly = True

        t = time.time()

        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = test.test_net(net, image, text_threshold, link_threshold, low_text, cuda,
                                                              poly, canvas_size, mag_ratio, show_time, refine_net)
        bbox_score = {}

        for box_num in range(len(bboxes)):
            key = str(det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key] = item

        output_masked_path = os.path.join(result_path, os.path.splitext(image_name)[0] + '_masked.jpg')
        cv2.imwrite(output_masked_path, score_text)
        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_path)

        print("elapsed time : {}s".format(time.time() - t))

        return str(bbox_score)
    except Exception as e:
        print('Cannot detect text regions(s) from {}'.format(image_path))
        print(e)
        return None
