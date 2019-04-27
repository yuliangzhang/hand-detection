import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import string_int_label_map_pb2
import os
import cv2
import imutils as im
import numpy as np
import math

from PIL import Image

# image_path = 'images/test3.png'
# image_path = 'images/test.jpg'
image_path = 'images/test2.jpeg'


# score threshold for showing bounding boxes.
score_thresh = 0.8
MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')
detection_graph = tf.Graph()
NUM_CLASSES = 1


# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))


            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            return p1, p2


    return None, None

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def _validate_label_map(label_map):
    """Checks if a label map is valid.

    Args:
      label_map: StringIntLabelMap to validate.

    Raises:
      ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    """Loads label map proto and returns categories list compatible with eval.

    This function loads a label map and returns a list of dicts, each of which
    has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    We only allow class into the list if its id-label_id_offset is
    between 0 (inclusive) and max_num_classes (exclusive).
    If there are several items mapping to the same id in the label map,
    we will only keep the first one in the categories list.

    Args:
      label_map: a StringIntLabelMapProto or None.  If None, a default categories
        list is created with max_num_classes categories.
      max_num_classes: maximum number of (consecutive) label indices to include.
      use_display_name: (boolean) choose whether to load 'display_name' field
        as category name.  If False or if the display_name field does not exist,
        uses 'name' field as category names instead.
    Returns:
      categories: a list of dictionaries representing all possible categories.
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def load_labelmap(path):
    """Loads label map proto.

    Args:
      path: path to StringIntLabelMap proto text file.
    Returns:
      a StringIntLabelMapProto
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map


def get_label_map_dict(label_map_path):
    """Reads a label map and returns a dictionary of label names to id.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.name] = item.id
    return label_map_dict


def process_image(img):
    CORRECTION_NEEDED = False
    # Define lower and upper bounds of skin areas in YCrCb colour space.
    lower = np.array([0, 139, 60], np.uint8)
    upper = np.array([255, 180, 127], np.uint8)
    # convert img into 300*x large
    # r = 300.0 / img.shape[1]
    # dim = (300, int(img.shape[0] * r))
    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    original = img.copy()

    # Extract skin areas from the image and apply thresholding
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    mask = cv2.inRange(ycrcb, lower, upper)
    skin = cv2.bitwise_and(ycrcb, ycrcb, mask=mask)
    _, black_and_white = cv2.threshold(mask, 127, 255, 0)

    # Find contours from the thresholded image
    _, contours, _ = cv2.findContours(black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the maximum contour. It is usually the hand.
    length = len(contours)
    maxArea = -1
    final_Contour = np.zeros(img.shape, np.uint8)
    # print(final_Contour)
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        largest_contour = contours[ci]

    # print(largest_contour)
    # Draw it on the image, in case you need to see the ellipse.
    cv2.drawContours(final_Contour, [largest_contour], 0, (0, 255, 0), 2)

    # Get the angle of inclination
    ellipse = _, _, angle = cv2.fitEllipse(largest_contour)

    # original = cv2.bitwise_and(original, original, mask=black_and_white)

    # Vertical adjustment correction
    '''
    This variable is used when the result of hand segmentation is upside down. Will change it to 0 or 180 to correct the actual angle.
    The issue arises because the angle is returned only between 0 and 180, rather than 360.
    '''
    vertical_adjustment_correction = 0
    if CORRECTION_NEEDED: vertical_adjustment_correction = 180

    # Rotate the image to get hand upright
    if angle >= 90:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction + 180 - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
        final_Contour = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
    else:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction - angle)
        final_Contour = im.rotate_bound(final_Contour, vertical_adjustment_correction - angle)

    original = cv2.bitwise_and(original, original, mask=black_and_white)
    # cv2.imshow('Extracted Hand', final_Contour)
    #cv2.imshow('Original image', original)

    # 求手掌中心
    # 参考至http://answers.opencv.org/question/180668/how-to-find-the-center-of-one-palm-in-the-picture/
    # 因为已经是黑白的，所以省略这一句
    # cv2.threshold(black_and_white, black_and_white, 200, 255, cv2.THRESH_BINARY)

    distance = cv2.distanceTransform(black_and_white, cv2.DIST_L2, 5, cv2.CV_32F)
    # Calculates the distance to the closest zero pixel for each pixel of the source image.
    # maxdist = 0
    # # rows,cols = img.shape
    # for i in range(distance.shape[0]):
    #     for j in range(distance.shape[1]):
    #         # 先扩展一下访问像素的 .at 的用法：
    #         # cv::mat的成员函数： .at(int y， int x)
    #         # 可以用来存取图像中对应坐标为（x，y）的元素坐标。
    #         # 但是在使用它时要注意，在编译期必须要已知图像的数据类型.
    #         # 这是因为cv::mat可以存放任意数据类型的元素。因此at方法的实现是用模板函数来实现的。
    #         dist = distance[i][j]
    #         if maxdist < dist:
    #             x = j
    #             y = i
    #             maxdist = dist
    # 得到手掌中心并画出最大内切圆
    final_img = original.copy()
    # cv2.circle(original, (x, y), maxdist, (255, 100, 255), 1, 8, 0)
    # half_slide = maxdist * math.cos(math.pi / 4)
    # # (left, right, top, bottom) = ((x - half_slide), (x + half_slide), (y - half_slide), (y + half_slide))
    # (left, right, top, bottom) = ((x - maxdist), (x + maxdist), (y - maxdist), (y + maxdist))
    # p1 = (int(left), int(top))
    # p2 = (int(right), int(bottom))
    # cv2.rectangle(original, p1, p2, (77, 255, 9), 1, 1)
    # final_img = final_img[int(top):int(bottom),int(left):int(right)]
    # cv2.imshow('palm image', original)
    return final_img



def image_sharp(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def palm_print_process(image):

    gray_org = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Image.fromarray(gray_org).show()

    binary_org = cv2.adaptiveThreshold(gray_org,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

    Image.fromarray(binary_org).show()



    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    gray = clahe.apply(gray_org)

    Image.fromarray(gray).show()

    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

    Image.fromarray(binary).show()

    gray_sharp = image_sharp(gray)

    Image.fromarray(gray_sharp).show()

    binary_sharp = cv2.adaptiveThreshold(gray_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    Image.fromarray(binary_sharp).show()






    # Image.fromarray(gray).show()
    #
    #
    # edges = cv2.Canny(gray, 60, 65, apertureSize=3)
    #
    # Image.fromarray(edges).show()
    #
    #
    # edges = cv2.bitwise_not(edges)
    #
    # Image.fromarray(edges).show()
    #
    # cv2.imwrite("palmlines.jpg", edges)
    # palmlines = cv2.imread("palmlines.jpg")
    # img = cv2.addWeighted(palmlines, 0.3, image, 0.7, 0)
    #
    # Image.fromarray(img).show()


def main():
    #read image
    cap = cv2.imread(image_path)
    #resize image
    r = 500.0 / cap.shape[1]
    dim = (500, int(cap.shape[0] * r))
    cap = cv2.resize(cap, dim, interpolation=cv2.INTER_AREA)

    im_width, im_height = (cap.shape[1], cap.shape[0])
    # max number of hands we want to detect/track
    num_hands_detect = 1
    image_np = cap
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
        quit(-1)

    # actual detection
    boxes, scores = detect_objects(
        image_np, detection_graph, sess)

    # draw bounding boxes
    p1, p2 = draw_box_on_image(
        num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

    if p1 is None:
        print('未检测到手掌，请重新尝试~')
        return

    palm_img = cap[p1[1]:p2[1], p1[0]:p2[0]]

    # Image.fromarray(image_np).show()
    # Image.fromarray(cv2.cvtColor(palm_img, cv2.COLOR_BGR2RGB)).show()


    palm_img = process_image(palm_img)



    # Image.fromarray(cv2.cvtColor(palm_img, cv2.COLOR_BGR2RGB)).show()

    palm_print_process(palm_img)

    # cv2.imshow('Hand Detection Result', cv2.cvtColor(
    #     image_np, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)


if __name__ == '__main__':
    # load label map
    label_map = load_labelmap(PATH_TO_LABELS)
    categories = convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = create_category_index(categories)
    detection_graph, sess = load_inference_graph()
    # while using imread , it will return a numpy array.
    main()
