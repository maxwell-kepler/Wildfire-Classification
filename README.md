# Wildfire Classification
This project utilizes deep learning concepts to classify wildfires using satellite images. To speed up the training, transfer learning was employed using the ResNet-50 model with a custom classifier replacing the output layer. Training was split into 2 distinct sections: first, freezing all convolutional layers except the classifier head, and secondly fine tuning layer 4 (The convolutional layer before output). The dataset was preprocessed to reflect the training data for ResNet-50, and has been augmented during training to improve classification. The model has reliable classification, with an accuracy of 98.8%. These results demonstrate the potential in machine learning to mitigate the risk of wildfires through quick and reliable identification.

## 📁 Project Files
```plaintext
Wildfire-Classification/ 
├── Wildfire_Classification.py       # Main script: model architecture, training, and fine-tuning 
├── Wildfire_Classification.ipynb    # Evaluation notebook: testing, metrics, and visualizations 
├── training_output.out              # Example output from the latest training run 
├── run.slurm                        # SLURM batch script for GPU-based training on TALC cluster
```
