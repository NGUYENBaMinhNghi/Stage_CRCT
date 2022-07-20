import numpy as np
from cartesio.easy import read_dataset, load_model, load_genome
from cartesio.dataset import DataReader, DataItem
from micromind.cv.conversion import bgr2hsv, gray2rgb, bgr2rgb
from micromind.io.image import imread_color, imread_grayscale, imread_tiff
from skimage.color import rgb2hed
from cartesio.fitness import FitnessIOU
from cartesio.plot import plot_mask, plot_markers, plot_watershed
from micromind.cv.image import split_channels
from cartesio.core.endpoint import Endpoint
from cartesio.endpoint import EndpointThreshold
from itertools import combinations
from cartesio.applications.segmentation import create_segmentation_model
import matplotlib.pyplot as plt
import cv2 as cv
from micromind.cv.image import image_normalize
import seaborn as sn
import pandas as pd
import os


class EndpointConvexHull(Endpoint):
    def __init__(self, threshold=1, exclude_holes=True):
        super().__init__("convex_hull", 1)
        self.threshold = threshold
        self.exclude_holes = exclude_holes

    def execute(self, entries):
        mask = entries[0].copy()
        mask[mask < self.threshold] = 0
        mask[mask >= self.threshold] = 255
        size = np.shape(mask)[0]
        mode = cv.RETR_LIST
        method = cv.CHAIN_APPROX_SIMPLE
        if self.exclude_holes :
            mode = cv.RETR_EXTERNAL
        contours, pas_important = cv.findContours(mask, mode, method)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv.convexHull(contours[i])
            hull_list.append(hull)
        img = np.zeros((size,size))
        cv.fillPoly(img, pts=hull_list, color=(255,255,255))
        return {"mask": img}



class EndpointPolygon(Endpoint):
    def __init__(self, coord_x, coord_y, threshold=1, exclude_holes=False):
        super().__init__("polygon", 1)
        self.threshold = threshold
        self.exclude_holes = exclude_holes
        self.x = coord_x
        self.y = coord_y

    def execute(self, entries):
        mask = entries[0].copy()
        mask[mask < self.threshold] = 0
        mask[mask >= self.threshold] = 255
        coord = np.array([self.x, self.y])
        mode = cv.RETR_LIST
        method = cv.CHAIN_APPROX_SIMPLE
        if self.exclude_holes :
            mode = cv.RETR_EXTERNAL
        contours, pas_important = cv.findContours(mask, mode, method)
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours[i][j] += coord
        return {"polygon": contours}

    
def evaluate_model(model_path, workspace, dataset, convex_hull=False):
    model = load_model(workspace + model_path)
    test_x = dataset.test_x
    test_y = dataset.test_y
    model.parser.endpoint = EndpointThreshold(4)
    if convex_hull:
         model.parser.endpoint = EndpointConvexHull(threshold=4)
    fitness = FitnessIOU()
    p, t = model.predict(test_x)
    return fitness.evaluate(test_y, [p])


def model_to_function(model_path, workspace):
    model = load_model(workspace + model_path)
    genome = load_genome(workspace + model_path)
    writer = GenomeToPython(model.parser.shape, model.parser.function_bundle)
    writer.to_code('create_mask', genome[1])
    return writer


def predict_model(model_path, workspace, dataset, convex_hull=False, show=False, save_im=False):
    model = load_model(workspace + model_path)
    model.parser.endpoint = EndpointThreshold(4)
    if convex_hull:
         model.parser.endpoint = EndpointConvexHull(threshold=4)
    visuals_test = dataset.testing.visuals
    test_x = dataset.test_x
    test_y = dataset.test_y
    p, t = model.predict(test_x)
    if show :
        for i in range(len(visuals_test)):
            mask = p[i]["mask"]
            plot_mask(visuals_test[i], mask.astype(np.uint8), gt = test_y[i][0].astype(np.uint8))
            if save_im:
                plt.savefig(f"image_mask_test_{i}.png")
    return p


def predict_model_polygon(model_path, workspace, test_x, coordx, coordy, exclu=False):
    model = load_model(workspace + model_path)
    model.parser.endpoint = EndpointPolygon(coordx, coordy, threshold=4, exclude_holes=exclu)
    p, t = model.predict(test_x)
    return p


def create_list_model_name(path_to_model_folder):
    files = os.listdir(path_to_model_folder)
    for i in range(len(files)):
        files[i] = files[i]+"/elite.json"
    return files

def create_model_list(path_list, workspace, convex_hull=False):
    model_list = []
    for i in path_list:
        model = load_model(workspace + i)
        model.parser.endpoint = EndpointThreshold(4)
        if convex_hull:
            model.parser.endpoint = EndpointConvexHull(threshold=4)
        model_list += [model]
    return model_list


def mean_masking(model_list, dataset, show = False):
    n = len(model_list)
    visuals_test = dataset.testing.visuals
    test_x = dataset.test_x
    test_y = dataset.test_y
    mean, t0 = model_list[0].predict(test_x)
    m = len(mean)
    for i in np.arange(1,n):
        p, t = model_list[i].predict(test_x)
        for j in range(m):
            mean[j]["mask"] += p[j]["mask"]
    for k in range(m):
        mean[k]["mask"] = mean[k]["mask"]/m
    if show :
        for l in range(len(visuals_test)):
            mask = mean[l]["mask"]
            plot_mask(visuals_test[l], mask.astype(np.uint8), gt = test_y[l][0].astype(np.uint8))
    return mean


def mean_masking2(model_list, dataset, show=False):
    y = []
    visuals_test = dataset.testing.visuals
    test_x = dataset.test_x
    test_y = dataset.test_y
    p_all = [model_list[mi].predict(test_x)[0] for mi in range(len(model_list))]
    for image in range(len(p_all[0])):
        p_image = np.array([image_normalize(p_all[mi][image]["mask"]) for mi in range(len(model_list))])
        p_image_mean = (p_image.mean(axis=0) * 255).astype(np.uint8) 
        y.append(p_image_mean)
    if show :
        for l in range(len(visuals_test)):
            mask = y[l]
            plot_mask(visuals_test[l], mask.astype(np.uint8), gt = test_y[l][0].astype(np.uint8))
    return y

def ensemble_masking(model_list, dataset, threshold=0.5, show=False, save_im=False):
    #It is similar to mean_masking but with threshold and different way to acces
    y = []
    visuals_test = dataset.testing.visuals
    test_x = dataset.test_x
    test_y = dataset.test_y
    p_all = [model_list[mi].predict(test_x)[0] for mi in range(len(model_list))]
    for image in range(len(p_all[0])):
        p_image = np.array([image_normalize(p_all[mi][image]["mask"]) for mi in range(len(model_list))])
        p_image_mean = p_image.mean(axis=0)
        p_image_mean[p_image_mean<threshold]=0
        p_image_mean[p_image_mean>=threshold]=255
        y.append(p_image_mean)
    if show :
        for l in range(len(visuals_test)):
            mask = y[l]
            plot_mask(visuals_test[l], mask.astype(np.uint8), gt = test_y[l][0].astype(np.uint8))
            if save_im:
                plt.savefig(f"image_ensemble_masking_test_{l}.png")
    return y

def ensemble_masking_polygon_1_patch(model_list, test_x, coordx, coordy, threshold=0.5, exclude_holes=True):
    coord = np.array([coordx, coordy])
    mode = cv.RETR_LIST
    method = cv.CHAIN_APPROX_SIMPLE
    if exclude_holes :
        mode = cv.RETR_EXTERNAL
    p_all = [model_list[mi].predict(test_x)[0] for mi in range(len(model_list))]
    p_image = np.array([image_normalize(p_all[mi][0]["mask"]) for mi in range(len(model_list))])
    p_image_mean = p_image.mean(axis=0)
    p_image_mean[p_image_mean<threshold]=0
    p_image_mean[p_image_mean>=threshold]=255
    p_image_mean = p_image_mean.astype(np.uint8)
    contours, pas_important = cv.findContours(p_image_mean, mode, method)
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours[i][j] += coord
    return contours
  

class ImageHEDReader(DataReader):
    def __init__(self):
        super().__init__("image", "hed")

    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_rgb = bgr2rgb(image_bgr)
        image_hed = rgb2hed(image_rgb)
        hed_channels0 = split_channels(image_hed)
        hed_channels = [(hed_channels0[j] * 255).astype(np.uint8) for j in range(len(hed_channels0))]
        return DataItem(hed_channels, image_hed.shape[:2], None, image_rgb)

    
class ImageHSV_HEDReader(DataReader):
    def __init__(self):
        super().__init__("image", "hsv_hed")

    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hsv = bgr2hsv(image_bgr)
        image_rgb = bgr2rgb(image_bgr)
        image_hed = rgb2hed(image_rgb)
        list_channels = []
        hsv_channels = split_channels(image_hsv)
        hed_channels0 = split_channels(image_hed)
        hed_channels = [(hed_channels0[j] * 255).astype(np.uint8) for j in range(len(hed_channels0))]
        for i in range(len(hsv_channels)):
            list_channels.append(hsv_channels[i])
        for i in range(len(hed_channels)):
            list_channels.append(hed_channels[i])
        return DataItem(list_channels, image_bgr.shape[:2], None, image_bgr)


class ImageRGB_HSV_HEDReader(DataReader):
    def __init__(self):
        super().__init__("image", "rgb_hsv_hed")

    def _read(self, filepath, shape=None):
        image_bgr = imread_color(filepath)
        image_hsv = bgr2hsv(image_bgr)
        image_rgb = bgr2rgb(image_bgr)
        image_hed = rgb2hed(image_rgb)
        list_channels = []
        hsv_channels = split_channels(image_hsv)
        rgb_channels = split_channels(image_bgr)
        hed_channels0 = split_channels(image_hed)
        hed_channels = [(hed_channels0[j] * 255).astype(np.uint8) for j in range(len(hed_channels0))]
        for i in range(len(rgb_channels)):
            list_channels.append(rgb_channels[i])
        for i in range(len(hsv_channels)):
            list_channels.append(hsv_channels[i])
        for i in range(len(hed_channels)):
            list_channels.append(hed_channels[i])
        return DataItem(list_channels, image_bgr.shape[:2], None, image_bgr)

    
def MC_crossvalidation(dataset, train_x, train_y, _lambda=5, generations=200, nb_nodes=35) :
    B = np.arange(0, len(train_x))
    MC_list = np.array(list(combinations(B, 1)))
    n = len(MC_list)
    score_MC = np.zeros(n)

    for i in range(n):
        x_test = []
        y_test = []
        x_train = []
        y_train = []
        for j in range(15):
            if (j in MC_list[i]):
                x_test += [train_x[j]]
                y_test += [train_y[j]]
            else:
                x_train += [train_x[j]]
                y_train += [train_y[j]]
        model = create_segmentation_model(generations, _lambda, inputs=dataset.inputs, nodes = nb_nodes)
        elite, history = model.fit(x_train, y_train)
        score_MC[i] = model.evaluate(x_test, y_test)[0]
    plt.boxplot(score_MC)
    return score_MC

def TuneCGP(mutation_node, mutation_output, dataset, _lambda=5, generations=200, nb_nodes=35, nb_model=5):
    a = len(mutation_node)
    b = len(mutation_output)
    train_x = dataset.train_x
    train_y = dataset.train_y
    test_x = dataset.test_x
    test_y = dataset.test_y
    score = 0
    Tune = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            score = 0
            for k in range(nb_model):
                model = create_segmentation_model(generations, _lambda, inputs=dataset.inputs, nodes = nb_nodes)
                elite, history = model.fit(train_x, train_y)
                score += model.evaluate(test_x, test_y)[0]
            Tune[i,j] = score/nb_model
    df = pd.DataFrame(Tune,index=mutation_node,columns=mutation_output)
    hm = sn.heatmap(data = df)
    plt.show()
    return Tune
    
#fonction pas très nécessaire 
#pour passer dans plot.mask il faut faire mask.astype(np.uint8)
def EnveloppeConvexe(mask, exclude_holes=True, threshold=4):
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255
    size = np.shape(mask)[0]
    mode = cv.RETR_LIST
    method = cv.CHAIN_APPROX_SIMPLE
    if exclude_holes :
        mode = cv.RETR_EXTERNAL
    contours, pas_important = cv.findContours(mask, mode, method)
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
    img = np.zeros((size,size))
    cv.fillPoly(img, pts=hull_list, color=(255,255,255))
    return img



