import numpy as np
import os
import albumentations as A
import cv2

# nuscenes 和 pickle 只有 get_nuscenes_set() 用到，懒加载避免在无 nuscenes 环境报错
def _lazy_nuscenes():
    import pickle
    import nuscenes
    from nuscenes.utils.splits import create_splits_scenes
    return pickle, nuscenes, create_splits_scenes

promptset_base = {
    "pedestrian": ["person", "pedestrian"],
    "barrier": ["barrier", "barricade"],
    "traffic_cone": ["traffic cone"],
    "bicycle": ["bicycle"],
    "bus": ["bus"],
    "car": ["car"],
    "construction vehicle": ["bulldozer", "excavator", "concrete mixer", "crane", "dump truck"],
    "motorcycle": ["motorcycle"],
    "trailer":["trailer", "semi trailer", "cargo container", "shipping container", "freight container"],
    "truck": ["truck"],
    "drivable surface": ["road"],
    "other_flat": ["curb", "traffic island", "traffic median"],
    "sidewalk": ["sidewalk"],
    "terrain": ["grass", "grassland", "lawn", "meadow", "turf", "sod"],
    "manmade": ["building", "wall", "pole", "awning"],
    "vegetation":[ "tree", "trunk", "tree trunk", "bush", "shrub", "plant", "flower", "woods"],
    "sky": ["sky"]
}

promptset_thing_base = {
    "pedestrian": ["person", "pedestrian"],
    "barrier": ["barrier", "barricade"],
    "traffic_cone": ["traffic cone"],
    "bicycle": ["bicycle"],
    "bus": ["bus"],
    "car": ["car"],
    "construction vehicle": ["bulldozer", "excavator", "concrete mixer", "crane", "dump truck"],
    "motorcycle": ["motorcycle"],
    "trailer":["trailer", "semi trailer", "cargo container", "shipping container", "freight container"],
    "truck": ["truck"],

}

promptset_stuff_base = {
    "drivable surface": ["road"],
    "other_flat": ["curb", "traffic island", "traffic median"],
    "sidewalk": ["sidewalk"],
    "terrain": ["grass", "grassland", "lawn", "meadow", "turf", "sod"],
    "manmade": ["building", "wall", "pole", "awning"],
    "vegetation":[ "tree", "trunk", "tree trunk", "bush", "shrub", "plant", "flower", "woods"],
    "sky": ["sky"]
}

promptset_semantic_kitti_base = {
	"car": ["car"],
	"bicycle": ["bicycle"],
	"motorcycle": ["motorcycle"],	
	"truck": ["truck"],
	"other-vehicle": ["trailer", "semi trailer", "cargo container", "shipping container", "freight container", "caravan", "bus", "bulldozer", "excavator", "concrete mixer", "crane", "dump truck", "train", "tram"], 
	"person": ["person", "pedestrian"],
	"bicyclist": ["bicyclist", "cyclist"],
    "motorcyclist": ["motorcyclist"],
    "road": ["road"],
    "parking": ["parking", "parking lot"],
	"sidewalk": ["sidewalk", "curb", "bike path", "walkway", "pavement", "footpath", "footway", "boardwalk", "driveway"], ## +
    "other-ground": ["water", "river", "lake", "watercourse", "waterway", "canal", "ditch", "rail track",
                    "traffic island", "traffic median", "median strip", "roadway median", "central reservation"], ## +
    "building": ["building", "house", "garage", "wall", "stairs", "railing", "awning", "roof", "bridge"], ## +
    "fence": ["fence", "barrier", "barricade"],
	"vegetation":[ "tree", "bush", "shrub", "plant", "flower"], 
	"trunk": ["tree trunk", "trunk", "woods"],
    "terrain": ["terrain", "grass", "grassland", "hill", "soil", "sand", "gravel", "lawn", "meadow", "garden","earth", "peeble", "rock"], ## +
	"pole": ["pole"],
	"traffic-sign": ["traffic sign"],
	"sky": ["sky"]
}


def get_nuscenes_set(split="val"):

   if split=="val":
       save_name = './nuscenes_validation_view_images.pkl'
   else:
       save_name = './nuscenes_train_view_images.pkl'

   pickle, nuscenes, create_splits_scenes = _lazy_nuscenes()

   if os.path.isfile(save_name):
       with open(save_name, 'rb') as f:
           view_images = pickle.load(f)
       return view_images

   cam_front, cam_front_right, cam_front_left = [], [], []
   cam_back, cam_back_right, cam_back_left = [], [], []

   phase_scenes = create_splits_scenes()[split]

   nusc = nuscenes.NuScenes(version="v1.0-trainval", dataroot="/datasets/nuscenes", verbose=True)

   for scene_idx in range(len(nusc.scene)):

       scene = nusc.scene[scene_idx]

       if scene["name"] in phase_scenes:

           current_sample_token = scene["first_sample_token"]

           while current_sample_token != "":
               current_sample = nusc.get("sample", current_sample_token)

               cam_front.append(nusc.get("sample_data", current_sample["data"]["CAM_FRONT"])["filename"][8:].split("/")[-1])
               cam_front_right.append(
                   nusc.get("sample_data", current_sample["data"]["CAM_FRONT_RIGHT"])["filename"][8:].split("/")[-1])
               cam_front_left.append(nusc.get("sample_data", current_sample["data"]["CAM_FRONT_LEFT"])["filename"][8:].split("/")[-1])
               cam_back.append(nusc.get("sample_data", current_sample["data"]["CAM_BACK"])["filename"][8:].split("/")[-1])
               cam_back_right.append(nusc.get("sample_data", current_sample["data"]["CAM_BACK_RIGHT"])["filename"][8:].split("/")[-1])
               cam_back_left.append(nusc.get("sample_data", current_sample["data"]["CAM_BACK_LEFT"])["filename"][8:].split("/")[-1])

               current_sample_token = current_sample["next"]

   final_dict = {"CAM_FRONT": cam_front, "CAM_FRONT_LEFT": cam_front_left, "CAM_FRONT_RIGHT": cam_front_right,
                 "CAM_BACK": cam_back, "CAM_BACK_LEFT": cam_back_left, "CAM_BACK_RIGHT": cam_back_right}
   print(f'DONE {len(final_dict)}')

   with open(save_name, 'wb') as f:
       pickle.dump(final_dict, f)

   return final_dict

def get_augmentation(aug_type):
        ## augmentations
    if aug_type=='color-jitter':
        augmentation = A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0, p=1.0)
        print(aug_type)
    elif aug_type=='hue-saturation':
        augmentation = A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
        print(aug_type)
    elif aug_type=='sharpen':
        augmentation = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), method='kernel',  p=1.0)
        print(aug_type)
    elif aug_type=='blur':
        augmentation = A.Blur(blur_limit=(3, 7), p=1.0)
        print(aug_type)
    elif aug_type=='auto-contrast':
        augmentation = A.AutoContrast(p=1.0)
        print(aug_type)
    elif aug_type=='horizontal-flip':
        augmentation = A.HorizontalFlip(p=1.0)
        print(aug_type)
    elif aug_type=='vertical-flip':
        augmentation = A.VerticalFlip(p=1.0)
        print(aug_type)
    elif aug_type=='chromatic-aberration':
        augmentation = A.ChromaticAberration(primary_distortion_limit=0.05,secondary_distortion_limit=0.1,mode='green_purple',interpolation=cv2.INTER_LINEAR,p=1.0)
    elif aug_type=='defocus':
        augmentation = A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4),p=1.0)
    elif aug_type=='emboss':
        augmentation = A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0)
    elif aug_type=='fancy-pca':
        augmentation = A.FancyPCA(alpha=0.1, p=1.0)
    elif aug_type=='gauss-noise':
        augmentation = A.GaussNoise(std_range=(0.1, 0.2), p=1.0)  # 10-20% of max value
    elif aug_type=='glass-blur':
        augmentation =  A.GlassBlur(sigma=0.7, max_delta=4, iterations=3, mode="fast", p=1.0)
    elif aug_type=='iso-noise':
        augmentation =  A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
    elif aug_type=='clahe':
        augmentation =  A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
    elif aug_type=='strong-1':
        print(aug_type)
        augmentation = A.Compose([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0, p=1.0),
                # A.HorizontalFlip(p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.Blur(blur_limit=(3, 7), p=1.0),
                A.ChromaticAberration(primary_distortion_limit=0.05,secondary_distortion_limit=0.1,mode='green_purple',interpolation=cv2.INTER_LINEAR,p=1.0),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                A.FancyPCA(alpha=0.1, p=1.0),
                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0), 
        ])
    elif aug_type=='random-brightness-contrast':
        print(f"aug type is {aug_type}")
        augmentation =  A.RandomBrightnessContrast(p=1.0)
    elif aug_type=='rgb-shift':
        augmentation =  A.RGBShift(p=1.0)
    elif aug_type=='random-fog':
        augmentation =  A.RandomFog(alpha_coef=0.1, p=1.0)
    elif aug_type=='random-gamma':
        augmentation = A.RandomGamma(p=1.0)
    elif aug_type=='random-rain':
        augmentation = A.RandomRain(p=1.0)
    elif aug_type=='random-shadow':
        augmentation = A.RandomShadow(p=1.0)
    elif aug_type=='random-snow':
        augmentation = A.RandomSnow(p=1.0)
    elif aug_type=='random-sun-flare':
        augmentation = A.RandomSunFlare(p=1.0)
    elif aug_type=='random-tone-curve':
        augmentation = A.RandomToneCurve(scale=0.2, per_channel=True, p=1.0)
    elif aug_type=='salt-and-pepper':
        augmentation = A.SaltAndPepper(p=1.0)
    elif aug_type=='shot-noise':
        augmentation = A.ShotNoise(p=1.0)
    elif aug_type=='solarize':
        augmentation = A.Solarize(p=1.0) 
    elif aug_type=='superpixels':
        augmentation =  A.Superpixels(p=1.0)
    elif aug_type=='to-gray':
        augmentation =  A.ToGray(method="weighted_average", p=1.0)

    return augmentation