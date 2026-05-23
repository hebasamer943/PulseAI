# Smart ECG Monitoring and Classification System using AI

A desktop-based AI system for ECG signal monitoring, real-time waveform visualization, beat classification, and post-session analysis.

## Features

- Real-time ECG waveform visualization
- ECG signal acquisition from ESP32 and AD8232 sensor
- CSV-based ECG playback and testing
- Beat classification using AI models
- Signal quality evaluation
- Heart rate monitoring
- Post-session analysis
- AI-assisted report generation
- PDF report export

## Technologies Used

- Python
- PySide6
- PyQtGraph
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- ESP32
- AD8232 ECG Sensor
- Arduino IDE

## Project Structure

```text
AI/              AI report generation module
Arduino Code/    ESP32 + AD8232 Arduino code
Dataset/         Dataset-related files or references
desktop_app/     Desktop application source code
Models/          Model loading instructions
sample_data/     Small ECG sample files for testing