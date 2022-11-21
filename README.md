# Unet Model

## Usage

- You need to load the model using the load_model function
- Get segmentation for a single photo using the get_segmentation function

## Learn Model

- Mark up photos. You can use a convenient markup site https://roboflow.com/.
- To train the model, you need to get ordinary photos and their mask, then divide them into test and validation.
- You can use the parsing_annotation_form_roboflow.py file to mark up the mask
- You need to put original photos and mask in different folders and add their path to config.py
- You can set the training parameters in the config.py file to start training.
