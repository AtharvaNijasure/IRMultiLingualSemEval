# IRMultiLingualSemEval

Most of the code is written by the author's of this report.
Please refer to CS646_Final_Report.pdf for more details.

# Instructions to Run the Project

## 1. Download the Data
- Download the required data to the `./data` folder.
- Steps to download:
  1. Access the data from Google Drive: [Google Drive Link](https://drive.google.com/drive/folders/1cg5VC4ULDGNzicUMVD2chxzMh7l9A6IE)
  2. Navigate to the `data` directory:
     ```bash
     cd data
     ```
  3. Place the downloaded data into the `data` folder.

## 2. Document Augmentation
- Run the following command:
  ```bash
  python expand_fact_check_docs.py


## Adjusting the Prompt and CSV Paths
- Modify the prompt and CSV file paths in the script depending on whether you're using **monolingual** or **cross-lingual** data.
- Use the `istart` and `iend` variables in the script to control the range of data generation.

## Running Baselines and Document Augmentation Experiments
1. Open and run the Jupyter notebook: `BuildIndexAndRunExperiments.ipynb`.
2. Update the following parameters in the notebook:
   - **CSV file paths**
   - **Data split**
   - **Task type**
3. Execute the notebook to obtain the results.

## Other Files
- The remaining files in the repository are helper files.