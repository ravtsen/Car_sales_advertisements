# Triple Ten Sprint 6 project - car sales advertisements - web application build.

## Project description

Car sales advertisements project **aim** is to analyse second hand car market.
[Web application] (https://github.com/ravtsen/Project_sprint_6/blob/main/app.py) performs *simple webscraping* of vehicles marketing data.
User Input Features allows to compare different models, compare car characteristics.    
There is also *market analyses* with *data visualization* in the app.

## Python libraries 

pandas, plotly, mumpy, streamlit, pillow

## Web application address:
https://vehicles-analytics.onrender.com/


## How to launch vehicals analytics app locally
For *launching an app locally* you need to save project folder to your device.

**Project repository tree**

├── README.md
├── app.py              
├── vehicles_us.csv
├── .gitignore
├── used_cars.jpeg
├── requirements.txt
└── notebooks
    └── EDA.ipynb
└── .streamlit
    └── config.toml

app.py - application
vehicles_us.csv - dataset
EDA.ipynb - exploratory data analysis 
config.toml - runs an application in server (headless) mode
requirements.txt - listing of all Python packages required to run the application

**Launching an app locally**
First, make sure streamlit is installed:
*pip install streamlit*
Streamlit application is defined in app.py. To run it locally, use the streamlit run command from the root of the project repository folder:
*streamlit run app.py*
The output should link to a URL that will host a webpage with the empty application. If you set the serverAddress to "0.0.0.0" in the configuration, you should enter http://0.0.0.0:10000/ in your browser to see the output.

### App user interface
Choose **data filters** from User Input Features on the left:
*Select model*
*Select car haracteristics*
*Select condition*
*Select type*
*Select fuel*
*Select mileage*

On the right there is car advertising data based on used filters.

Below you can find 4 buttons with *market analyses* and *data visualization*:
*Car price dependence on odometer readings and car age*
*Car types popularity*
*Car price dependence on condition*
*Transmission impact on market distribution and prices*
