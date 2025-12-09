Steps of Implementation:
1) Data Collection and Pre-processing-
Currency note images were collected and organized into separate folders for training, validation, and testing. The dataset included annotated regions such as the watermark window, security thread, and number panel. All images were resized and prepared according to the YOLO training structure.

2) Feature Annotation for YOLO Model-
Using the collected images, bounding-box annotations were created for the three key features essential in verifying note authenticity. These annotations were stored in YOLO label format and linked in the ‘data.yaml’ file specifying train, validation, and class names.

3) Training YOLO Detection Model-
The YOLO model was trained using the annotated dataset for 50 epochs with a batch size of 32. The training process produced ‘best.pt’, the optimized model used for detecting currency features in uploaded images.

4) Dataset Augmentation Using GAN-
A DCGAN architecture was implemented to expand the feature dataset. Starting from neural noise, the Generator produced synthetic patches of currency features while the Discriminator learned to differentiate real and generated samples. This process increased the dataset to 5000 images, ensuring robust feature learning.

5) Generating Positive and Negative Pairs for Siamese Network-
YOLO-generated feature crops of genuine and fake notes were combined to build pair datasets (pair_img1.npy, pair_img2.npy, pair_labels.npy). Positive pairs contained two matching genuine features, while negative pairs combined real and fake feature samples. These pairs were essential for training the Siamese similarity model.

6) Training the Siamese Similarity Network-
A custom Siamese network was created using a shared CNN encoder and an L1 distance layer. The model was trained to predict similarity between two images using binary cross-entropy loss for 15 epochs. The final trained file was saved in .keras format for deployment.

7) Flask Application Development-
A Flask server was built to integrate both YOLO and Siamese models. The web interface allowed users to upload currency note images through an HTML form. The uploaded images were saved temporarily on the server for processing.

8) Real-time Prediction Workflow-
The Flask backend applied the YOLO model to detect key security features from the uploaded image. Each cropped feature was passed to the Siamese model, which compared it with reference genuine features and returned similarity scores. Based on these scores, the system declared the currency note as REAL or FAKE. The resulting bounding-box image and prediction were displayed back on the webpage.

9) Final Deployment & Output Visualization-
The final system generated annotated output images showing YOLO’s detected features and displayed the Siamese similarity decision to the user. The integration enabled an end-to-end automated fake currency detection pipeline accessible through a simple web interface.

Dataset Link- https://www.kaggle.com/datasets/gowthamreddyuppunuri/indian-currency-notes-used-for-yolov5 
