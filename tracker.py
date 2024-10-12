from __future__ import division
import time
import os

import torch
from torch.autograd import Variable
import cv2
import typer
from typing_extensions import Annotated

from objectdetect.util import *
from objectdetect.darknet import Darknet
from objectdetect.preprocess import *

app = typer.Typer()


@app.command()
def detect(images: Annotated[str, typer.Argument(help="Image file or a directory containg images to perform detection upon.")] = "imgs",
           output_loc: Annotated[str, typer.Argument(
               help="Directory for outputting detection results.")] = "output",
           batch_size: Annotated[int, typer.Argument(help="Batch size")] = 1,
           confidence: Annotated[float, typer.Argument(
               help="Confidence for filtering detections [0, 1]")] = 0.5,
           nms_thresh: Annotated[float, typer.Argument(
               help="NMS Threshold")] = 0.4,
           cfg: Annotated[str, typer.Argument(
               help="Configuration file location")] = "cfg/yolov3.cfg",
           weights: Annotated[str, typer.Argument(
               help="The location of the weights file.")] = "data/yolov3.weights",
           reso: Annotated[int, typer.Argument(help="The resolution of the input image, should be a number divisible by 32.")] = 416):

    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80  # For COCO
    classes = load_classes("data/coco.names")

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(cfg)
    model.load_weights(weights)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
    try:
        imlist = [os.path.join(os.path.realpath('.'), images, img)
                  for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(os.path.join(os.path.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(output_loc):
        try:
            os.makedirs(output_loc)
        except FileExistsError:
            pass

    load_batch = time.time()
    batches = list(
        map(prepare_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size,
                                                              len(im_batches))])) for i in range(num_batches)]

    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(
            prediction, confidence, num_classes, nms_conf=nms_thresh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        # transform the atribute from index in batch to index in imlist
        prediction[:, 0] += i*batch_size

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    class_load = time.time()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1))/2
    output[:, 1:5] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    colors = palette
    list(map(lambda x: write_image(x, orig_ims, classes, colors), output))
    output_files = map(
        lambda x: f"{output_loc}/det_{x.split("\\")[-1]}", imlist)
    for (file, img) in zip(list(output_files), orig_ims):
        if not cv2.imwrite(file, img):
            print(f"Failed to write output for file path: {file}")

    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format(
        "Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format(
        "Output Processing", output_recast - class_load))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - output_recast))
    print("{:25s}: {:2.3f}".format(
        "Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
