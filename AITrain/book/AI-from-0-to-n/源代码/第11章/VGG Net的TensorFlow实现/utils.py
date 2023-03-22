import skimage
import skimage.io
import skimage.transform
import numpy as np


def load_image(path):

    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    pred = np.argsort(prob)[::-1]


    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))

    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

