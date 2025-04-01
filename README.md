# MoviePosterTranslation

## Environment
Create Anaconda environment

conda install -c conda-forge teseract

## Structure
Download data folder from Onedrive and put it right under the most outside directory.

Make results folder that has two subfolders inside called 'blurred' and 'final_image' inside.


## Setting up your env file
First you need to install dotenv by running `pip install python-dotenv` in your terminal. Then you need to create a .env file in the root directory of the project and add the following variables
```
API_KEY=your_api_key
TESSERACT_PATH=your_tesseract_path
```

## Execution

### vision_model
Change the folder path for pytesseract