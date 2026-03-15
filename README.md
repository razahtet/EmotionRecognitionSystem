# Real-Time Speech Emotion Recognition System
A deep learning system built to detect human emotions from live speech audio. The system transforms raw audio into spectrogram representations for multi-class emotion classification using Convolutional Neural Networks (CNNs) and recurrent networks to predict emotional states.

## Technical Approach
* **Preprocessing:** Converting raw audio data into spectrograms to visualize sound patterns.
* **Modeling:** Applying supervised classification via CNNs for pattern recognition and potentially KNN for speaker similarity clustering.
* **Inference:** Developing a live web application for real-time detection via microphone input and confidence scoring.
* **Optimization:** Using mathematical modeling to maximize feature extraction accuracy.

## Datasets
* **[Crema-D (Crowd-sourced Emotional Multimodal Actors Dataset)](https://www.kaggle.com/datasets/ejlok1/cremad):**
  * **Stats:** 7,442 original clips from 91 actors (48 male, 43 female).
  * **Diversity:** Ages 20–74 across various ethnicities (African American, Asian, Caucasian, Hispanic, and Unspecified).
  * **Access:** Hosted through Hugging Face.
  * **Cleaning:** Got rid of any files that were silent, corrupted, or too short.
  * **Link:** [razahtet/crema-d-cleaned](https://huggingface.co/datasets/razahtet/crema-d-cleaned)
    #### How to access the cleaned data in code:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("razahtet/crema-d-cleaned")
    ```
## Future Datasets
* **[AESDD (Acted Emotional Speech Dynamic Database)](https://github.com/SuperKogito/SER-datasets):** Emotional Speech Dataset that contains acted emotional utterances for mulitple categories including happiness, anger, sadness, and neutrality.

## Project Roadmap (In Progress)
* **Data Science Pipeline:** Complete collection, cleaning, and feature engineering.
* **Infrastructure:** Structured database for audio files, extracted features, and model predictions.
* **Model Development:** Training, validating, and performance-testing the CNN-based classifier.
* **Deployment:** API development for real-time predictions and a live web interface.

## Collaborators
* Raza (Project Lead)<!--:CNN architecture design, implementation, and neural network optimization for spectrogram analysis.-->
* Alex <!--:NLP speech pattern analysis and supervised classification model development.-->
* Aneesh <!-- Mathematical optimization of audio feature extraction and hyperparameter tuning.-->
* Matthew <!-- Adaptive system design and user experience optimization for applications in gaming environments.-->
