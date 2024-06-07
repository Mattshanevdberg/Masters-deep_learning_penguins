import subprocess
import os
import shutil

# list of paths to data
PARENT_PATH_RAW_DATA = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/raw/OpenImagesV7/'
K5ParentList = ['k5_data_split_list_1', 'k5_data_split_list_2','k5_data_split_list_3','k5_data_split_list_4','k5_data_split_list_5']
#K5ParentList = ['k5_data_split_list_2','k5_data_split_list_3','k5_data_split_list_4','k5_data_split_list_5']
K5_YOLO_DATA_PATH = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_K5/'
list_of_dir_to_clear = ['YoloV8_dataset_OI7_K5/images/train', 'YoloV8_dataset_OI7_K5/images/val', 'YoloV8_dataset_OI7_K5/labels/train', 'YoloV8_dataset_OI7_K5/labels/val']
list_of_model_names = ['yolov8n.yaml', 'yolov8n.pt', 'yolov8s.yaml', 'yolov8s.pt', 'yolov8m.yaml', 'yolov8m.pt']
script_name = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_gen.py'
script_name_val = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_gen.py'
cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'

### Pre-training loading of data into correct format and directories for k5 experiments

# load ids into list 
def load_file_to_list(file_path):
    """
    loads a text file to a list with each entry on a new line becoming a new entry in the list.

    :param file_path: Path to the file where the list should be saved.
    :return list of data from file
    """
    # Open the file for writing
    lst = []
    with open(file_path, 'r') as file:
        # Write each item on a new line
        for line in file:
            lst.append(line.strip())
    return lst

def perform_train_val_split(K5_curr_Split):
    '''
    load chosen split to val and rest of the ids to a train list. Creating the test train split for the current k-5 experiment.
    param: K5_curr_Split: name of the file that contains the ids of the current k5 split that is to be used as val 
    return: K5_val_id_list: list of the validation ids
    return: K5_train_id_list: list of the train ids
    '''
    # load chosen split to val and rest of the ids to a train list
    K5_val_id_list = load_file_to_list(PARENT_PATH_RAW_DATA+K5_curr_Split)
    K5_train_id_list = []
    for split in K5ParentList:
        if split != K5_curr_Split:
            K5_train_id_list.extend(load_file_to_list(PARENT_PATH_RAW_DATA+split))
    return K5_val_id_list, K5_train_id_list

def clear_directory(dir_path):
    """
    Removes all files and subdirectories within the specified directory without deleting the directory itself.

    :param dir_path: Path to the directory to clear.
    """
    # Check if the directory exists
    if not os.path.isdir(dir_path):
        print("The specified path is not a directory.")
        return
    print(f'clearing {dir_path}')
    # Iterate over each item in the directory
    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)  # Full path to the item

        # Check if the item is a file or directory and remove it
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # Remove the file or link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory and all its contents

def clear_directories(list_of_dir_to_clear):
    '''
    Clears all the directories in the list.
    param: list_of_dir_to_clear: list of all the directories to clear (the path should be from K5_YOLO_DATA_PATH to the directory) 
    '''
    for dir in list_of_dir_to_clear:
        dir_to_clear = K5_YOLO_DATA_PATH+dir
        # clear dir
        clear_directory(dir_to_clear)

def copy_images_based_on_ids_in_list(list, source_dir, destination_dir):
    """
    Copies images from source directory to destination directory based on IDs in list.
    
    :param text_file_path: list containing IDs.
    :param source_dir: Directory where images are stored (images should be named as <id>.jpg).
    :param destination_dir: Directory to which images will be copied.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)


    for id_ in list:
        # Construct the filename and the paths
        filename = f"{id_}.jpg"
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)

        # Check if the file exists and copy it to the destination directory
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            #print(f"Copied {filename} to {destination_dir}")
        else:
            print(f"File {filename} not found in {source_dir}")

def copy_annotation_based_on_ids_in_list(list, source_dir, destination_dir):
    """
    Copies images from source directory to destination directory based on IDs in list.
    
    :param text_file_path: list containing IDs.
    :param source_dir: Directory where images are stored (images should be named as <id>.jpg).
    :param destination_dir: Directory to which images will be copied.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)


    for id_ in list:
        # Construct the filename and the paths
        filename = f"{id_}.txt"
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)

        # Check if the file exists and copy it to the destination directory
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            #print(f"Copied {filename} to {destination_dir}")
        else:
            print(f"File {filename} not found in {source_dir}")

def move_images_and_annotations_to_correct_folders(list_of_dir):
    '''
    Copy the images and annotations to the correct folders according to the list of train and val ids supplied
    param: list_of_dir: list of the yolo image and annotation directories that images and annotation will be copied to (same as list of dir to clear)
    '''
    # path to all annotations and images
    images_path = PARENT_PATH_RAW_DATA+'OpenImageV7_raw_images'
    annotations_path = PARENT_PATH_RAW_DATA+'OI7_Yolo_annotations_NG'

    # step through all the directories in the list of directory
    for dir in list_of_dir:
        dir_to_copy_to = K5_YOLO_DATA_PATH+dir
        # copy images to the desired directory
        if 'images' in dir:
                if 'train' in dir:
                    copy_images_based_on_ids_in_list(K5_train_id_list, images_path, dir_to_copy_to)
                else:
                    copy_images_based_on_ids_in_list(K5_val_id_list, images_path, dir_to_copy_to)
        # copy the annotations to the desired dir
        else:
                if 'train' in dir:
                    copy_annotation_based_on_ids_in_list(K5_train_id_list, annotations_path, dir_to_copy_to)
                else:
                    copy_annotation_based_on_ids_in_list(K5_val_id_list, annotations_path, dir_to_copy_to)


### Training 
def run_script(script_name, list_of_args):
    '''
    Run the python script whose path is being passed to this function. required to kill the process before each run (saving GPU memory)
    param: script_name: path to the python script
    param: list_of_args: a list of arguments that are being passed to the function in the python script.
    Current arguments: [model being run, path to config file, name of run to be saved]
    '''
    # Call the second script
    result = subprocess.run(['python', script_name] + list_of_args, capture_output=True, text=True)
    
    # Print output and error if any
    print("Output:", result.stdout)
    #print("Errors:", result.stderr)

    # Check the return code
    if result.returncode == 0:
        print("Script executed successfully")
    else:
        print("Script failed with return code", result.returncode)


# Call the function
# model_name = args[0], cfg_path = args[2], run_name = args[3]

# for loop to step through all the different k-5 combinations
for K5_curr_Split in K5ParentList:
    #load the current split into a train and val id list
    K5_val_id_list, K5_train_id_list = perform_train_val_split(K5_curr_Split)
    print('split lists')
    print(len(K5_val_id_list), len(K5_train_id_list))

    # clear the directories of any previous data from previous splits so that you can then load the new data
    clear_directories(list_of_dir_to_clear)
    print('cleared dir')

    # move the images and annotations into the correct directories based on the current split
    move_images_and_annotations_to_correct_folders(list_of_dir_to_clear)
    print('moved images and anno to dirs')

    for model_name in list_of_model_names:
        run_name = f'{K5_curr_Split}_{model_name}'
        
        list_of_args = [model_name, cfg_path, run_name]

        print(f'run model {run_name}')
        #print(f'list of args: {list_of_args}')
        #print(script_name)

        run_script(script_name, list_of_args)






#model = 'yolov8n.pt'
#cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'
#run_name = 'train_testRun_nanoPt_2_savejson'
#list_of_args = [model, cfg_path, run_name]
#run_script('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/detection_experiments_gen.py', list_of_args=list_of_args)

