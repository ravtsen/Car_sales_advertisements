# Car sales advertisements - web application build

## Project description

Car sales advertisements project **aim** is to analyse second hand car market.<br />
[Web application](https://github.com/ravtsen/Project_sprint_6/blob/main/app.py) performs *simple webscraping* of vehicles marketing data.
User Input Features allows to compare different models, compare car characteristics.    
There is also *market analyses* with *data visualization* in the app.

## Python libraries 

* Pandas<br />
* Plotly<br /> 
* Numpy<br />
* Streamlit<br />
* Pillow<br />

## [Link to web application](https://vehicles-analytics.onrender.com/)

## How to launch vehicals analytics app locally
For *launching an app locally* you need to save project folder to your device.

**Project repository tree**
<br />
├── README.md<br />
├── app.py<br />
├── vehicles_us.csv<br />
├── .gitignore<br />
├── used_cars.jpeg<br />
├── requirements.txt<br />
└── notebooks<br />
&nbsp;&nbsp;&nbsp;&nbsp;└── EDA.ipynb<br />
└── .streamlit<br />
&nbsp;&nbsp;&nbsp;&nbsp;└── config.toml<br />


* **app.py** - application<br />
* **vehicles_us.csv** - dataset<br />
* **EDA.ipynb** - exploratory data analysis<br />
* **config.toml** - runs an application in server (headless) mode<br />
* **requirements.txt** - listing of all Python packages required to run the application<br />

### Launching an app locally

First, make sure streamlit is installed:<br />
*pip install streamlit*<br />
Streamlit application is defined in app.py. To run it locally, use the streamlit run command from the root of the project repository folder:<br />
*streamlit run app.py*<br />
The output should link to a URL that will host a webpage with the empty application. If you set the serverAddress to "0.0.0.0" in the configuration, you should enter http://0.0.0.0:10000/ in your browser to see the output.

### App user interface

Choose **data filters** from User Input Features on the left:<br />

* *Select model*<br />
* *Select car haracteristics*<br />
* *Select condition*<br />
* *Select type*<br />
* *Select fuel*<br />
* *Select mileage*<br />

On the right there is car advertising data based on used filters.

Below you can find 4 buttons with *market analyses* and *data visualization*:<br />
* *Car price dependence on odometer readings and car age*<br />
* *Car types popularity*<br />
* *Car price dependence on condition*<br />
* *Transmission impact on market distribution and prices*<br />
