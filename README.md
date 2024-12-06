# Threats to Internal Validity in Business Process Simulation (BPS) Evaluations

## Overview

This repository contains data and scripts associated with the paper *"Threats to Internal Validity in Business Process Simulation (BPS) Evaluations."* The research explores how systematic errors introduced during various stages of Business Process Simulation (BPS) can distort evaluation results and lead to incorrect conclusions about the quality of a simulation model.

## Abstract

Business Process Simulation (BPS) is commonly used to conduct what-if analyses and assess the impact of changes to processes and their underlying information systems. In a typical data-driven BPS approach, a simulation model is derived from an event log and evaluated through a five-step method: log extraction, log splitting, model discovery, process simulation, and evaluation.

This repository provides the necessary files and scripts to replicate the empirical examples based on commonly used event logs, demonstrating the five systematic errors that can distort evaluation outcomes and emphasizing the importance of careful decision-making when evaluating BPS models.

### Identified Threats

1. **Concept Drift**: Occurs when the extracted event log contains data on different versions of a process, affecting model evaluation.
2. **Fading-in and Fading-out Phases**: Partial traces in the event log can misrepresent the actual process dynamics.
3. **Fading-in and Fading-out During Log Splitting**: Misrepresentation of active cases near the log splitting point.
4. **Data Leakage**: Retention of traces across training and test logs may lead to overestimation of model accuracy.
5. **Warm-up and Cool-down Phases**: Initial and final conditions in simulated logs can distort evaluation results, especially for dynamic parameters like waiting times.

## Getting Started

### Prerequisites

- Python 3.11.8
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/bps-validity-threats.git
    cd bps-validity-threats
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Navigate to the `notebooks` directory.
2. Run the provided Jupyter notebooks to replicate the evaluation and visualize the threats.
3. Use the adapted version of AgentSimulator to replicate the simulations: [AgentSimulator_WuCd](https://github.com/robert-l-b/AgentSimulator_WuCd)

## Notebooks

The `notebooks` folder contains the following Jupyter notebooks:

- **[`t1_concept_drift.ipynb`](notebooks/t1_concept_drift.ipynb)**
- **[`t2_3_extraction_splitting.ipynb`](notebooks/t2_3_extraction_splitting.ipynb)**
- **[`t4_data_leakage.ipynb`](notebooks/t4_data_leakage.ipynb)**
- **[`t5_warm_up_cool_down.ipynb`](notebooks/t5_warm_up_cool_down.ipynb)**
- **[`post_hoc_analysis.ipynb`](notebooks/post_hoc_analysis.ipynb)**
- **[`simulation_evaluation.ipynb`](notebooks/simulation_evaluation.ipynb)**

## Data Description

The `data` directory can be retrieved here: [Google Drive - Data Directory](https://drive.google.com/drive/folders/1pshLeqwjHTZLZMuM8KxBgAndHv5mE0oP?usp=share_link)

It contains:

- **`full_logs`**: Raw event logs used for simulation.
- **`input_data`**: Preprocessed logs split into training and test datasets.
- **`simulated_data`**: Simulated event logs from different simulation scenarios.

## License

This project is licensed under the [MIT License](LICENSE). 

## Acknowledgments

This repository is part of ongoing research on Business Process Simulation (BPS) and is based on empirical examples from publicly available event logs.
