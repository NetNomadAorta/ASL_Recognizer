import json

import cv2
import os
import torch
from torchvision import models
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
# from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from playsound import playsound


# User parameters
# SAVE_NAME       = "ASL_Letters.model"
# DATASET_PATH    = "./Training_Data/" + SAVE_NAME.split(".model",1)[0] +"/"
MIN_SCORE       = 0.7
IMAGE_SIZE      = 300
SHOULD_SCREENSHOT = False
SCREENSHOT_ITERATION = 50
N_FRAMES        = 10


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# dataset_path = DATASET_PATH

#load classes
# coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
# categories = coco.cats
# n_classes_1 = len(categories.keys())
# categories

# f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
# data = json.load(f)
# n_classes_1 = len(data["categories"])
# classes_1 = [i['name'] for i in data["categories"]]


# classes_1 = [i[1]['name'] for i in categories.items()]


# lets load the faster rcnn model
# Mobile Net
model_1 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# Faster RCNN
# model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
#                                                  min_size=IMAGE_SIZE,
#                                                  max_size=IMAGE_SIZE*3
#                                                     )
# Faster RCNN
# model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
# model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# Load model
if torch.cuda.is_available():
    map_loc_name = lambda storage, loc: storage.cuda()
else:
    map_loc_name = 'cpu'

# if os.path.isfile(SAVE_NAME):
#     checkpoint = torch.load(SAVE_NAME, map_location=map_loc_name)
#     model_1.load_state_dict(checkpoint)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

color_list = ['green', 'magenta', 'turquoise', 'red', 'green', 'orange', 'yellow', 'cyan', 'lime']



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# For recording video
# video = VideoWriter('webcam.avi', VideoWriter_fourcc(*'MJPG'), 10.0, (int(frame.shape[1]), int(frame.shape[0])) )

transforms_1 = A.Compose([
    # A.Resize(int(frame.shape[0]/2), int(frame.shape[1]/2)),
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()
ii = 0
tenScale = 10

while rval:
    cv2.imshow("preview", frame)
    # cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)
    rval, frame = vc.read()

    # Saves screenshot for training later
    if SHOULD_SCREENSHOT:
        if ii % SCREENSHOT_ITERATION == 0:
            cv2.imwrite("Images/{}.jpg".format( int(time.time() - 1674000000 )), frame)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]

    if ii % N_FRAMES == 0: # Inferences every n frames
        with torch.no_grad():
            prediction_1 = model_1([(transformed_image/255).to(device)])
            pred_1 = prediction_1[0]

    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].tolist()

    # labels_found = [str(round(die_scores[index]*100)) + "% - " + str(classes_1[class_index])
    #                 for index, class_index in enumerate(die_class_indexes)]
    
    predicted_image = draw_bounding_boxes(transformed_image,
        boxes = dieCoordinates,
        # labels = [classes_1[i] for i in die_class_indexes], 
        # labels = labels_found, # SHOWS SCORE AND INDEX IN LABEL
        width = 1,
        colors = ['purple' for i in die_class_indexes],
        font="arial.ttf",
        font_size=20
        )

    predicted_image_cv2 = predicted_image.permute(1, 2, 0).contiguous().numpy()
    predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)

    # for coordinate_index, coordinate in enumerate(dieCoordinates):dieCoordinates
    #     start_point = (int(coordinate[0]), int(coordinate[1]))
    #     # end_point = ( int(coordinate[2]), int(coordinate[3]) )
    #     color = (255, 0, 255)
    #     # thickness = 3
    #     # cv2.rectangle(predicted_image_cv2, start_point, end_point, color, thickness)
    #
    #     start_point_text = (start_point[0], max(start_point[1] - 5, 0))
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     fontScale = 1.0
    #     thickness = 1
    #
    #     # Draws background text
    #     # cv2.putText(predicted_image_cv2, labels_found[coordinate_index],
    #     #             start_point_text, font, fontScale, (0,0,0), thickness + 3)
    #     #
    #     # # Draws foreground text
    #     # cv2.putText(predicted_image_cv2, labels_found[coordinate_index],
    #     #             start_point_text, font, fontScale, color, thickness)

    # Can comment out - this is for testing
    # save_image((predicted_image/255), "image.jpg")
    
    # Changes image back to a cv2 friendly format
    # frame = predicted_image.permute(1,2,0).contiguous().numpy()
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = predicted_image_cv2

    if len(dieCoordinates) > 0 and ii % (N_FRAMES*3) == 0:
        cv2.imwrite("Images/{}.jpg".format(int(time.time() - 1674000000)), frame)

        # for playing note.wav file
        playsound("Do_It.wav")
        print('playing sound using  playsound')

    # Write frame to the video file
    # video.write(frame)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  ", round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()


# Release web camera stream
vc.release()
cv2.destroyWindow("preview")

# Release video output file stream
# video.release()