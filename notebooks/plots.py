import json
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_map50_scores(data):
    model_names = ["yolov8m.pt", "yolov8m.yaml", "yolov8s.pt", "yolov8s.yaml", "yolov8n.pt", "yolov8n.yaml"]
    splits = ["split1", "split2", "split3", "split4", "split5"]

    mAP50_data = {model: [] for model in model_names}

    for split in splits:
        for model in model_names:
            mAP50_data[model].append(data[split]['models'][model]['mAP50'])

    return mAP50_data

def plot_box_and_whisker(mAP50_data, output_file_path):
    model_names = list(mAP50_data.keys())
    values = [mAP50_data[model] for model in model_names]
    # Define more descriptive labels for the models
    descriptive_labels = {
        "yolov8n.pt": "Yolov8 Nano\n(Pretrained)",
        "yolov8s.pt": "Yolov8 Small\n(Pretrained)",
        "yolov8m.pt": "Yolov8 Medium\n(Pretrained)",
        "yolov8n.yaml": "Yolov8 Nano\n(Rand Init)",
        "yolov8s.yaml": "Yolov8 Small\n(Rand Init)",
        "yolov8m.yaml": "Yolov8 Medium\n(Rand Init)"
    }
    
    # Map the original model names to the descriptive labels
    descriptive_model_names = [descriptive_labels[model] for model in model_names]


    plt.figure(figsize=(12, 10))
    plt.boxplot(values, labels=descriptive_model_names, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'),
                flierprops=dict(marker='o', color='red', alpha=0.5))
    
    # Add title and labels to the plot
    plt.title('Box and whisker diagram of the k-fold cross validation\nof various Yolov8 models mAP-50 scores on the OI7 dataset', fontsize=22)
    plt.xlabel('Models', fontsize=20)
    plt.ylabel('mAP-50 Scores', fontsize=20)
    plt.xticks(fontsize=18, rotation=45, ha='right')
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to include the bottom of the y-axis labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_file_path)
    plt.show()

def extract_map50_scores_and_params(data):
    """
    Extract mAP-50 scores and parameters from the JSON data for each model and split.
    
    Parameters:
    data (dict): The JSON data containing model evaluation results.
    
    Returns:
    dict: A dictionary where keys are model names and values are lists of (mAP-50 score, parameters, speed) tuples for each split.
    """
    # List of model names to extract
    model_names = ["yolov8m.pt", "yolov8m.yaml", "yolov8s.pt", "yolov8s.yaml", "yolov8n.pt", "yolov8n.yaml"]
    # List of split names
    splits = ["split1", "split2", "split3", "split4", "split5"]

    # Initialize a dictionary to hold mAP-50 scores, parameters, and inference speed for each model
    model_data = {model: [] for model in model_names}

    # Loop through each split and extract mAP-50 scores, parameters, and speed for each model
    for split in splits:
        for model in model_names:
            model_info = data[split]['models'][model]
            map50 = model_info['mAP50']
            params = model_info['number of params'] / 1e6  # Convert to millions
            speed = model_info['inference speed (ms)']
            model_data[model].append((map50, params, speed))

    return model_data

def plot_yolo_map_vs_params_and_speed(model_data, output_file_path):
    """
    Create a plot of mAP-50 scores vs parameters and inference speed for YOLOv8 models and save it as a PDF.
    
    Parameters:
    model_data (dict): A dictionary where keys are model names and values are lists of (mAP-50 score, parameters, speed) tuples.
    output_file_path (str): The path to save the output PDF file.
    """
    # Extract data for plotting
    models = list(model_data.keys())
    map50_scores = [model_data[model] for model in models]
    
    # Separate the data
    map50_nano_pt = [x[0] for x in model_data["yolov8n.pt"]]
    params_nano_pt = [x[1] for x in model_data["yolov8n.pt"]]
    speed_nano_pt = [x[2] for x in model_data["yolov8n.pt"]]

    map50_small_pt = [x[0] for x in model_data["yolov8s.pt"]]
    params_small_pt = [x[1] for x in model_data["yolov8s.pt"]]
    speed_small_pt = [x[2] for x in model_data["yolov8s.pt"]]

    map50_medium_pt = [x[0] for x in model_data["yolov8m.pt"]]
    params_medium_pt = [x[1] for x in model_data["yolov8m.pt"]]
    speed_medium_pt = [x[2] for x in model_data["yolov8m.pt"]]

    map50_nano_yaml = [x[0] for x in model_data["yolov8n.yaml"]]
    params_nano_yaml = [x[1] for x in model_data["yolov8n.yaml"]]
    speed_nano_yaml = [x[2] for x in model_data["yolov8n.yaml"]]

    map50_small_yaml = [x[0] for x in model_data["yolov8s.yaml"]]
    params_small_yaml = [x[1] for x in model_data["yolov8s.yaml"]]
    speed_small_yaml = [x[2] for x in model_data["yolov8s.yaml"]]

    map50_medium_yaml = [x[0] for x in model_data["yolov8m.yaml"]]
    params_medium_yaml = [x[1] for x in model_data["yolov8m.yaml"]]
    speed_medium_yaml = [x[2] for x in model_data["yolov8m.yaml"]]
    
    # Create the plot for mAP-50 vs Parameters
    plt.figure(figsize=(12, 8))
    
    plt.scatter(params_nano_pt, map50_nano_pt, label='Yolov8 Nano (Pretrained)', color='#030b7c')
    plt.scatter(params_small_pt, map50_small_pt, label='Yolov8 Small (Pretrained)', color='#298932')
    plt.scatter(params_medium_pt, map50_medium_pt, label='Yolov8 medium (Pretrained)', color='#a10a0a')

    plt.scatter(params_nano_yaml, map50_nano_yaml, label='Yolov8 Nano (Rand Init)', color='#0691b9')
    plt.scatter(params_small_yaml, map50_small_yaml, label='Yolov8 Small (Rand Init)', color='#43c611')
    plt.scatter(params_medium_yaml, map50_medium_yaml, label='Yolov8 Medium (Rand Init)', color='#ef5325')
    
    plt.title('mAP-50 on the IO7 Dataset vs Parameters for YOLOv8 Models', fontsize=22)
    plt.xlabel('Parameters (Millions)', fontsize=20)
    plt.xlim(left=0)
    plt.ylabel('mAP-50 Score', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file_path.replace('.pdf', '_params.pdf'))
    
    # Create the plot for mAP-50 vs Inference Speed
    plt.figure(figsize=(12, 8))
    
    plt.scatter(speed_nano_pt, map50_nano_pt, label='Yolov8 Nano (Pretrained)', color='#030b7c')
    plt.scatter(speed_small_pt, map50_small_pt, label='Yolov8 Small (Pretrained)', color='#298932')
    plt.scatter(speed_medium_pt, map50_medium_pt, label='Yolov8 medium (Pretrained)', color='#a10a0a')

    plt.scatter(speed_nano_yaml, map50_nano_yaml, label='Yolov8 Nano (Rand Init)', color='#0691b9')
    plt.scatter(speed_small_yaml, map50_small_yaml, label='Yolov8 Small (Rand Init)', color='#43c611')
    plt.scatter(speed_medium_yaml, map50_medium_yaml, label='Yolov8 Medium (Rand Init)', color='#ef5325')
    
    plt.title('mAP-50 on the IO7 Dataset vs Inference Speed for YOLOv8 Models', fontsize=22)
    plt.xlabel('Inference Speed (ms)', fontsize=20)
    plt.xlim(left=0)
    plt.ylabel('mAP-50 Score', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file_path.replace('.pdf', '_speed.pdf'))
    
    plt.show()

if __name__ == "__main__":
    # Set the input and output file paths here
    input_json_file = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/k-5_val_results.json'
    output_pdf_file = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/k-5_val_results_box_and_whisker.pdf'
    output_pdf_file_scatter = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/k-5_val_results_scatter.pdf'
    

    # Load data, extract mAP-50 scores, and plot
    data = load_json(input_json_file)
    #mAP50_data = extract_map50_scores(data)
    #plot_box_and_whisker(mAP50_data, output_pdf_file)
    model_data = extract_map50_scores_and_params(data)
    plot_yolo_map_vs_params_and_speed(model_data, output_pdf_file)
