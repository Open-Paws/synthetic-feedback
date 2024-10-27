# Synthetic Feedback Data

## Objective
This script generates synthetic feedback to augment real human feedback for AI model training, addressing two key challenges:

1. **Diversity Enhancement**
While real human feedback is crucial, resource limitations make it challenging to gather data from the full spectrum of human cultural and demographic backgrounds needed for developing an inclusive AI system. The script helps bridge this gap by generating synthetic feedback from a diverse range of simulated human perspectives.

2. **Nonhuman Representation**
Since we cannot directly collect feedback from nonhuman animals about content that affects their lives, the script creates synthetic feedback that attempts to represent their perspectives. This helps ensure their interests are considered in the model training process.

## How It Works

The program operates by:

- Creating synthetic profiles (both human and nonhuman) with randomized characteristics including:
  - For humans: cultural background, education, beliefs, values, and psychometric traits
  - For nonhumans: species type, living situation, and circumstances
- Using these profiles to evaluate content in Label Studio tasks
- Generating feedback that reflects each profile's unique perspective
- Running this process iteratively to build a large, diverse feedback dataset

This synthetic data augments, rather than replaces, real human feedback, with the goal of creating a more comprehensive and inclusive training dataset that considers both human diversity and nonhuman perspectives.

## Usage

This script is currently being hosted on [Google Colab](https://colab.research.google.com/drive/1P12107jmzjMoKGL4hW-ux6xV2rW152de?usp=sharing#scrollTo=q8-tS2ptxM1m) without automatic integration with this Github repo. I aim to keep this Github repository manually updated with the Google Colab for version control purposes and in case we want to move it to a more dedicated service in the future (e.g., Google Cloud Run).
