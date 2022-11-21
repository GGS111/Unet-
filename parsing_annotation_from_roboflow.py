import json
import numpy as np
import cv2

def create_mask(path_to_anno, path_to_mask):
    '''
    path_to_anno - Path to annotation from Roboflow
    path_to_mask - Path to the folder where the mask will be added
    '''

    f = open (path_to_anno, errors='ignore')
    
    # Reading from file
    data = json.loads(f.read())

    areas = {}
    for area in data['annotations']:
        if area['image_id'] not in areas:
            areas[area['image_id']] = area['segmentation']
        else:
            areas[area['image_id']] += area['segmentation']

    for index, val in enumerate(data['images']):
        blank = np.zeros((400, 400))
        name = val['file_name'] 
        for area in areas[index]:
            area = np.array(area)
            area = area.reshape(int((area.shape[0])/2), 2).astype(int)
            blank = cv2.fillPoly(blank, pts=[area], color=(255, 255, 255))
        cv2.imwrite('{}{}_mask.png'.format(path_to_mask, name[:-4]), blank)

if __name__ == "__main__":
    create_mask('learn_model/test/train/_annotations.coco.json', 'learn_model/test/mask/')