# iRIS: Interactive Reverse Image Search

This readme serves as a manual for setting up the RelTR Scene Graph Generation model and then setting up the iRIS dashboard for reverse image search using the generated scene graph for a given input query image.

### Virtual Environment

If you do not have Python virtualenv installed, please run the following command to install virtualenv:

```pip install virtualenv``` or ```pip3 install virtualenv``` or ```python3 -m pip install virtualenv```

Setup the virtualenv by running the following commands:

```
python3 -m virtualenv .iris_env
source .iris_env/bin/activate
```

Install the necessary requirements using the following command:

```pip install -r requirements.txt```

### RelTR Setup

Download the [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under 
```
./sgg_model/ckpt/checkpoint0149.pth
```

Note that the RelTR model codebase has been added here in this repository for user's convenience. We have cited their work for this project. Please visit this [link](https://github.com/yrcong/RelTR) for their original code for generating Scene graphs.

### Search Space Setup

Our search system doesn't require the user to download the entire image dataset for Visual Genome for their search process. However, you would need to download and store the Image Metadata, and the Scene Graph JSON files to start your search process. Please follow the following steps to download and place the required JSON files.

Download the Image Metadata using this [link](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)

Download the Scene Graph JSON data using this [link](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip)

Unzip and move the two files to the ```data``` directory using the following commands. Replace the path to the downloaded data to your current working directory containing the JSON files. You would also need to replace the path to iRIS project directory below.

```
unzip {path_to_downloaded_data}/image_data.json.zip
mv {path_to_downloaded_data}/image_data.json {project_directory}/data
unzip {path_to_downloaded_data}/relationships.json.zip
mv {path_to_downloaded_data}/relationships.json {project_directory}/data
```

### Using the iRIS Dashboard

You first need to host the application using the following command. Please ensure that your virtual environment is activated before running the application.

```python3 iris_app.py```

Access the app in your browser by visiting this URL: ```http://localhost:8050/```

You have now setup the iRIS dashboard for your reverse image search process. Add your input image query by dragging and dropping your search image or by browsing it through your files. 

Give the RelTR model a few seconds to predict the Scene Graph for your input query image and it would present a Scene Graph for your input image on the dashboard shortly. The user can now begin their search process by selecting nodes or edges on the displayed scene graph. Their selected entities will be highlighted in the input image as well. The user will be displayed with dropdowns to display lower confidence scored relationship edges as well. Another dropdown will help the user to highlight specific relationships in blue within the selected query image. They can also display generic or more specific labels by moving the Hypernym slider below the input query image. 

Their search results are displayed at the bottom of the page as thumbnails of matching search results when the user selects a node and edge. 

### Authors
Aniruddha Prashant Deshpande
Mahek Mishra
Tanvi Kaple