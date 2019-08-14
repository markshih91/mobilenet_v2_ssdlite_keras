from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd


# model config
batch_size = 32
image_size = (300, 300, 3)
n_classes = 80
mode = 'inference'
l2_regularization = 0.0005
min_scale = 0.1 #None
max_scale = 0.9 #None
scales = None #[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None #[8, 16, 32, 64, 100, 300]
offsets = None #[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.01
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False

model = mobilenet_v2_ssd(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes)

# 2: Load the trained weights into the model.
weights_path = '../pretrained_weights/ssdlite_coco_loss-4.8205_val_loss-4.1873.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = '../experiments/imgs/000004.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(image_size[0], image_size[1]))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

plt.figure(figsize=(20, 12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / image_size[0]
    ymin = box[3] * orig_images[0].shape[0] / image_size[1]
    xmax = box[4] * orig_images[0].shape[1] / image_size[1]
    ymax = box[5] * orig_images[0].shape[0] / image_size[0]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})