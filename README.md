# Animal Pose Estimation Project
this is currently a placeholder to be updated

## Documentation
Project Structure: https://docs.google.com/document/d/1Zcc5bP5jzIF-3HsQOtWlXn6cRTWTmjbp5I2xviMNezw/edit?usp=sharing

## Introduction
This project aims to develop a machine learning pipeline for multiple animal pose estimation. It incorporates a detection module using YOLOv8, a segmentation module, and a pose estimation module. The project is developed initially in Jupyter notebooks for prototyping and visualization, followed by object-oriented programming for deployment on Raspberry Pi.

## Installation
Ensure you have Python 3.8+ installed. Clone this repository and install required dependencies:
git clone https://github.com/yourusername/animal-pose-estimation.git
cd animal-pose-estimation
pip install -r requirements.txt


## Usage
Start by exploring the Jupyter notebooks in the `notebooks/` directory for data exploration and initial model testing. For structured development, see the `src/` directory.

To run a detection example:
python src/detection/detector.py --image_path path/to/your/image.jpg

## Development
Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` for contribution guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the YOLOv8 team for the detection model.
- Dataset provided by [Dataset Source].

