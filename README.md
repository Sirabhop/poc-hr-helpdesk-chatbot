# POC-HR-Helpdesk-Chatbot

This project is a proof-of-concept (POC) for an HR Helpdesk Chatbot designed to assist in answering frequently asked HR-related questions.
The chatbot leverages machine learning models and integrates with external data sources to provide accurate and efficient responses.

## Folder Structure

The project repository is organized as follows:

```
POC-HR-HELPDESK-CHATBOT/
|
├── bin/               # Executable files and scripts
│   └── dev.ipynb      # Jupyter Notebook for development and testing
|
├── credential/        # Directory for storing API keys and authentication files
│   └── ktb-complaint-center-poc-<id>.json  # Example credential file
|
├── data/              # Contains datasets and input data files
│   └── HR Helpdesk_Q&A Chatbot.xlsx  # Excel file with training data or FAQs
|
├── model/             # Directory for machine learning models
│   └── gemini.py       # Code to interact with the Gemini model API
|
├── module/            # Contains reusable Python modules
│   ├── memory.py       # Memory management for chatbot state
│   ├── util.py         # Utility functions for the project
│
├── etl.py             # ETL (Extract, Transform, Load) pipeline for data processing
├── main.py            # Main script to run the chatbot application
├── .gitignore         # Git ignore file specifying files to exclude from version control
├── README.md          # Project documentation (this file)
└── venv/              # Virtual environment for dependency management
    └── (Python environment files)
```

## Description of Key Components

### 1. `bin/`
This folder contains development and testing scripts. It includes:
- **`dev.ipynb`**: A Jupyter Notebook used for experimenting with model and data.

### 2. `credential/`
This directory stores the authentication files required for external API integrations.
- **`ktb-complaint-center-poc-<id>.json`**: Example authentication file for secure API communication.

### 3. `data/`
Includes datasets or any external input files.
- **`HR Helpdesk_Q&A Chatbot.xlsx`**: This file may contain frequently asked questions (FAQs) and their corresponding answers to train or validate the chatbot.

### 4. `model/`
Houses machine learning models and associated scripts.
- **`gemini.py`**: A script that interacts with the Gemini model for AI-driven chatbot responses.

### 5. `module/`
Contains reusable modules that encapsulate specific functionalities.
- **`memory.py`**: Handles memory and context management for the chatbot.
- **`util.py`**: Provides utility functions such as data formatting, logging, etc.

### 6. Root Files
- **`etl.py`**: Manages the ETL pipeline to preprocess and load data into the chatbot system.
- **`main.py`**: The entry point to execute and run the chatbot application.
- **`.gitignore`**: Specifies files and folders to exclude from version control (e.g., virtual environment, credentials).
- **`README.md`**: This documentation file.

### 7. `venv/`
This folder contains the virtual environment setup for Python dependencies. Ensure you activate the environment before running the project:
```bash
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate   # Windows
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd POC-HR-HELPDESK-CHATBOT
   ```

2. **Set Up Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # MacOS/Linux
   # For Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Credentials**
   Place your API credential files in the `credential/` directory.

5. **Run the Application**
   Execute the main chatbot script:
   ```bash
   python main.py
   ```