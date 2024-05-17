# Dino Game Reinforcement Learning Project

## Overview

This repository is dedicated to solving the Google Chrome Dino game using reinforcement learning techniques. By leveraging advanced machine learning models and architectures, we provide a technical solution that allows a trained agent to play and navigate the game efficiently.

## Technical Approach

1. **Object Detection**: We use YoloV8 with fine-tuning to detect objects within the game's images. YoloV8 was selected for its training speed, crucial for the iterative nature of reinforcement learning.

2. **Feature Extraction**: From the object detection process, features such as the distance to the nearest obstacle, its height, and its bounding box are extracted. Additionally, the altitude of the Dino and its position in the previous frame are also considered. This detailed information is crucial for making real-time decisions in the game. The use of a multi-layer perceptron architecture aids in rapid inference, a key factor in the time-sensitive environment of the game.

3. **Deep Q-Network (DQN)**: The implementation includes enhancements like Prioritized Replay Buffer and Noisy Layers for effective exploration, along with a Dueling network architecture to separately assess the value of states and the advantage of actions.

## Installation

To install and run the project, follow these steps:

### Prerequisites
- An Anaconda distribution of Python installed on your machine. CUDA with a Nvidia GPU for tensorRT.

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/peduajo/DinoRL.git
   cd DinoRL
   ```

2. Create a Conda environment using the provided `environments.yml` file:
   ```bash
   conda env create -f environments.yml
   ```

3. Activate the newly created environment:
   ```bash
   conda activate dino_rl
   ```

## Training

To start training the DQN model, ensure that your computer’s internet connection is turned off (to prevent unwanted updates or interruptions), and run the following command:

   ```bash
   python train_dqn.py
   ```

## Contribution

Contributions are welcome. Please submit pull requests or open an issue if you have suggestions or find bugs. Let's make the Dino game AI better together!