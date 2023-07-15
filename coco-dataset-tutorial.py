#!/usr/bin/env python
# coding: utf-8

# # **A step-by-step tutorial for using the COCO dataset in computer vision research.**
# You can find the comprehensive tutorial in my **[blog post](https://armanasq.github.io/datasets/coco-datset/)**
# 
# 
# ## Introduction
# 
# Welcome to this comprehensive tutorial on the COCO (Common Objects in Context) dataset! 
# 
# The COCO dataset is a widely recognized benchmark in the field of computer vision and serves as a valuable resource for various tasks, including object detection, segmentation, and captioning.
# 
# In this tutorial, we will delve into the details of the COCO dataset, exploring its purpose, structure, and practical applications. We will cover a range of topics, from installing the required libraries to analyzing the dataset's category distribution and visualizing images with annotations.
# 
# By the end of this tutorial, you will have a thorough understanding of the COCO dataset and be equipped with the necessary knowledge to leverage its power for your computer vision research and applications.
# 
# ## Prerequisites
# 
# To follow along with this tutorial, you should have a basic understanding of computer vision concepts and Python programming. Familiarity with libraries such as `numpy`, `matplotlib`, and `seaborn` will be beneficial.
# 
# ## Dataset Overview
# 
# The COCO dataset is a large-scale dataset designed to facilitate research in object recognition, segmentation, and captioning. It contains a diverse collection of images, each annotated with extensive information about the objects present.
# 
# The key features of the COCO dataset are as follows:
# 
# - **Object Categories**: The dataset consists of 80 distinct object categories, including common objects like "person," "car," "dog," and more. Each object category is associated with a unique ID.
# 
# - **Annotations**: For each image in the dataset, precise annotations are provided, including bounding box coordinates, segmentation masks, and category labels. These annotations allow for detailed object localization and understanding.
# 
# - **Complexity**: The COCO dataset contains images with varying levels of complexity, including multiple objects, occlusions, and diverse backgrounds. This complexity makes it an excellent benchmark for evaluating computer vision algorithms.
# 
# ## Installing Required Libraries
# 
# Before we dive into working with the COCO dataset, it's essential to ensure that the necessary libraries are installed. Some of the key libraries we will be using are:
# 
# - `pycocotools`: This library provides a Python API for accessing and manipulating the COCO dataset.
# 
# - `matplotlib` and `seaborn`: These libraries offer powerful visualization capabilities, allowing us to display images, plots, and annotations.
# 
# - `numpy`: This fundamental library is used for numerical operations and data manipulation.
# 
# Ensure that you have these libraries installed by following the installation instructions specific to your system and environment.
# 
# ## Loading COCO Dataset
# 
# To work with the COCO dataset, you need to obtain the dataset files and organize them on your local machine. The dataset is available for download from the COCO website, and it consists of two main components:
# 
# - **Images**: The COCO dataset comprises a vast collection of images, grouped into different sets (train, validation, and test). These images form the core of the dataset, providing visual data for various computer vision tasks.
# 
# - **Annotations**: Alongside the images, the COCO dataset includes detailed annotations for each image. These annotations specify object boundaries, segmentation masks, and category labels. They serve as ground truth data for training and evaluation purposes.
# 
# Once you have downloaded the dataset, ensure that you have organized the image files and annotations into appropriate directories on your local machine. This organization will make it easier to access and work with the data using the COCO API.
# 
# ## Initializing COCO API
# 
# To interact with the COCO dataset, we will utilize the COCO API, a Python library that provides convenient access to the dataset's images and annotations.
# 
# To get started, you need to initialize the COCO API by specifying the paths to the annotation file and the image directory. The annotation file contains the metadata and annotations for the images, while the image directory stores the actual image files.
# 
# By initializing the COCO API, you can leverage its functionalities to retrieve specific images, annotations, and category information for further analysis and visualization.
# 
# ## Exploring Category Information
# 
# One crucial aspect of working with the COCO dataset is understanding the object categories it encompasses. Each object in the dataset is assigned a specific category, and it is essential to have an overview of the available categories and their distribution.
# 
# To explore the category information, we can utilize the COCO API to retrieve category IDs, names, and counts. This information allows us to understand the diversity of object categories in the dataset and their relative frequencies.
# 
# Analyzing the category distribution is helpful for gaining insights into the dataset's composition and can guide us in formulating strategies for training and evaluating computer vision models.
# 
# ## Loading and Displaying Images
# 
# A fundamental task when working with the COCO dataset is loading and displaying images. Images provide visual context for the objects and annotations present in the dataset.
# 
# By utilizing the COCO API, we can load images based on their IDs and retrieve their corresponding annotations. Once the images are loaded, we can display them using libraries such as `matplotlib`. This visual representation helps us get a visual understanding of the dataset and the objects it contains.
# 
# ## Visualizing Category Distribution
# 
# To gain a comprehensive understanding of the COCO dataset, it is beneficial to visualize the distribution of object categories. This visualization provides insights into the relative frequencies of different object categories, enabling us to identify dominant categories and potentially imbalanced distributions.
# 
# By leveraging libraries like `matplotlib` and `seaborn`, we can create visualizations such as bar plots and pie charts to represent the category distribution. These visualizations aid in identifying patterns, biases, and potential challenges associated with specific categories.
# 
# ## Filtering Images by Category
# 
# The COCO dataset is vast, containing images with a diverse range of objects. To focus our analysis on specific object categories of interest, we can filter the images based on those categories.
# 
# By specifying the desired object categories, we can retrieve the image IDs that contain objects belonging to those categories. This filtering allows us to narrow down the dataset and focus on the specific objects or categories we want to study.
# 
# ## Displaying Filtered Images with Annotations
# 
# Once we have filtered the images based on specific object categories, we can display the filtered images along with their annotations. This step involves retrieving the annotations corresponding to the filtered images and visualizing them using tools like `matplotlib`.
# 
# The annotations provide valuable information about object boundaries, segmentation masks, and category labels. By overlaying these annotations on the images, we can gain a comprehensive understanding of the object localization and characteristics within the filtered subset.
# 
# ---
# 
# 
# The COCO dataset continues to be a valuable resource for the computer vision community, driving advancements in object detection, segmentation, and captioning. By mastering the techniques and concepts covered in this tutorial, you are well-equipped to embark on your own computer vision projects using the COCO dataset.
# 
# Happy exploring and discovering the rich world of objects in the COCO dataset!

# 
# Let's start with the first step, which is installing the pycocotools package.
# 
# # **Step 1: Installing pycocotools**
# 
# The pycocotools package is a Python library that provides convenient tools for working with the COCO dataset. It includes APIs for reading and manipulating the annotation files, which are commonly used in computer vision tasks. To install pycocotools, you can follow these steps:
# 
# 1. Open a terminal or command prompt.
# 2. Ensure that you have Python and pip installed on your system.
# 3. Execute the following command:
# 
# ```
# !pip install pycocotools
# ```
# 
# This command will download and install the pycocotools package from the Python Package Index (PyPI). Depending on your system configuration, you may need to prefix the command with `sudo` to install the package with administrator privileges.
# 
# Once the installation is complete, you can proceed to the next steps of the tutorial.

# In[2]:


get_ipython().system('pip install pycocotools')


# Install the Prerequisites

# In[2]:


get_ipython().system('pip install scipy==1.11 numpy==1.22.0')


# # **Step 2: Importing Required Libraries**
# 
# Before working with the COCO dataset, you need to import certain libraries that will be used for data visualization and processing. In this step, we import the following libraries:
# 
# - matplotlib: A widely used library for creating visualizations in Python.
# - matplotlib.pyplot: A sub-module of matplotlib that provides a MATLAB-like interface for creating plots and visualizations.
# - matplotlib.patches: A submodule of matplotlib that provides various patch classes, such as rectangles, circles, and polygons, which can be used to draw bounding boxes on images.
# - matplotlib.colors: A submodule of matplotlib that provides functions for working with colors.
# - seaborn: A library built on top of matplotlib, which provides additional plotting capabilities and enhances the visual aesthetics.
# - numpy: A fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices.
# 
# Additionally, we import the COCO class from the pycocotools module, which allows us to read and manipulate the COCO dataset's annotation files. To import these libraries and the COCO class, include the following lines of code at the beginning of your script:
# 
# By importing these libraries, you'll have access to various functions and classes that will be used throughout the tutorial for visualizing and working with the COCO dataset.

# In[3]:


import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import numpy as np

from pycocotools.coco import COCO


# # **Step 3: Setting Up COCO Dataset and Initializing API**
# 
# In this step, we set up the COCO dataset and initialize the COCO API for working with instance annotations. Follow these instructions to accomplish this:
# 
# 1. Specify the data directory (`dataDir`): Set the variable `dataDir` to the path of your COCO dataset directory. In this example, the COCO dataset is assumed to be located in the directory `/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/`. Modify this path to match the location of your dataset.
# 
# 2. Specify the data type (`dataType`): Set the variable `dataType` to the specific data split you want to work with. In this example, we set it to `'val2014'`, which corresponds to the validation split of the COCO dataset. You can change this value to `'train2014'` or `'test2014'` based on your requirements.
# 
# 3. Specify the annotation file (`annFile`): Use the `annFile` variable to define the path to the COCO dataset's instance annotation file. This file contains the bounding box and category information for each annotated object. In this example, the annotation file path is set as `'annotations/instances_val2014.json'` within the data directory specified earlier. Modify this path accordingly if your dataset follows a different structure.
# 
# 4. Specify the image directory (`imageDir`): Set the `imageDir` variable to the path of the directory containing the dataset images. In this example, the image directory path is set as `'images/val2014/'` within the data directory. Adjust this path if your dataset images are stored in a different location or follow a different naming convention.
# 
# 5. Initialize the COCO API: Create an instance of the COCO class by passing the annotation file path (`annFile`) to the constructor. This initializes the COCO API and allows you to access various methods for interacting with the dataset. In this example, we initialize it as `coco = COCO(annFile)`.
# 
# By completing these steps, you have set up the COCO dataset and initialized the COCO API, enabling you to access and manipulate the instance annotations and images in the dataset.

# In[4]:


dataDir='/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/'
dataType='val2014'
annFile='{}annotations/instances_{}.json'.format(dataDir,dataType)
imageDir = '{}/images/{}/'.format(dataDir, dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)


# # **Step 4: Loading Categories from COCO Dataset**
# 
# In this step, we load the categories from the COCO dataset based on the provided category IDs. Follow these instructions to load the categories:
# 
# 1. Specify the category ID(s): Set the variable `ids` to the category ID(s) you want to load from the COCO dataset. In this example, `ids` is set to `1`, indicating that we want to load the category with ID 1. You can modify this value to load one or multiple categories based on your requirements.
# 
# 2. Load the categories: Use the `loadCats(ids)` method of the COCO API to load the categories. Pass the `ids` variable as an argument to the method. This method returns a list of category dictionaries containing information such as the category ID, name, and supercategory. In this example, we load the categories and store them in the `cats` variable using the line `cats = coco.loadCats(ids=ids)`.
# 
# 3. Print the categories: Use the `print()` function to display the loaded categories. In this example, we print the `cats` variable, which contains the category information for the provided IDs. This allows you to verify that the correct categories have been loaded.
# 
# 
# After executing this code, you will see the information about the loaded category/categories printed in the console or output area.

# In[5]:


# Load categories for the given ids
ids = 1
cats = coco.loadCats(ids=ids)
print(cats)


# Also, you can print all the categories by the following code:

# In[7]:


category_ids = coco.getCatIds()
num_categories = len(category_ids)
print('number of categories: ',num_categories)
for ids in category_ids:
    cats = coco.loadCats(ids=ids)
    print(cats)


# # **Step 5: Loading Images from COCO Dataset**
# 
# In this step, we load images from the COCO dataset based on the provided image IDs. Follow these instructions to load the images:
# 
# 1. Get the image IDs: Use the `getImgIds()` method of the COCO API to retrieve a list of all image IDs in the dataset. Store this list in the `image_ids` variable.
# 
# 2. Specify the image ID: Set the variable `image_id` to the specific image ID you want to load from the COCO dataset. In this example, we set it as `image_ids[0]`, which corresponds to the first image ID in the list. You can change this line to display a different image by modifying the index value or by selecting a specific image ID from the `image_ids` list.
# 
# 3. Load image information: Use the `loadImgs(image_id)` method of the COCO API to load the image information for the specified image ID. This method returns a list of dictionaries containing details such as the image ID, file name, width, height, and more. In this example, we load the image information and store it in the `image_info` variable using the line `image_info = coco.loadImgs(image_id)`.
# 
# 4. Print the image information: Use the `print()` function to display the loaded image information. In this example, we print the `image_info` variable, which contains the details of the image corresponding to the provided ID.
# 
# 
# 
# After executing this code, you will see the information about the loaded image printed in the console or output area. This includes details such as the image ID, file name, width, height, and more.

# In[13]:


# Load images for the given ids
image_ids = coco.getImgIds()
image_id = image_ids[0]  # Change this line to display a different image
image_info = coco.loadImgs(image_id)
print(image_info)


# # **Step 6: Loading Annotations from COCO Dataset**
# 
# In this step, we load annotations from the COCO dataset based on the provided image IDs. Follow these instructions to load the annotations:
# 
# 1. Get the annotation IDs: Use the `getAnnIds(imgIds=image_id)` method of the COCO API to retrieve a list of annotation IDs for a specific image. Pass the `image_id` variable as an argument to the method. Store the list of annotation IDs in the `annotation_ids` variable.
# 
# 2. Load annotations: Use the `loadAnns(annotation_ids)` method of the COCO API to load the annotations corresponding to the provided annotation IDs. This method returns a list of annotation dictionaries containing information such as the annotation ID, category ID, bounding box coordinates, and segmentation mask. In this example, we load the annotations and store them in the `annotations` variable using the line `annotations = coco.loadAnns(annotation_ids)`.
# 
# 3. Print the annotations: Use the `print()` function to display the loaded annotations. In this example, we print the `annotations` variable, which contains the details of the annotations for the given image ID.
# 
# After executing this code, you will see the information about the loaded annotations printed in the console or output area. This includes details such as the annotation ID, category ID, bounding box coordinates, and segmentation mask for each annotation associated with the provided image ID.

# In[15]:


# Load annotations for the given ids
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)
print(annotations)


# # **Step 7: Filtering Category IDs based on Given Conditions**
# 
# In this step, we filter the category IDs from the COCO dataset based on certain conditions. Follow these instructions to filter the category IDs:
# 
# 1. Specify the filter conditions: Set the variable `filterClasses` to a list of category names or classes that you want to filter. In this example, the `filterClasses` list contains the strings `'laptop'`, `'tv'`, and `'cell phone'`. Modify this list to include the desired category names.
# 
# 2. Get the category IDs: Use the `getCatIds(catNms=filterClasses)` method of the COCO API to retrieve the category IDs that satisfy the given filter conditions. Pass the `filterClasses` list as an argument to the method. This method returns a list of category IDs that correspond to the provided category names. In this example, we fetch the category IDs and store them in the `catIds` variable using the line `catIds = coco.getCatIds(catNms=filterClasses)`.
# 
# 3. Print the category IDs: Use the `print()` function to display the filtered category IDs. In this example, we print the `catIds` variable, which contains the category IDs that satisfy the given filter conditions.
# 
# 
# After executing this code, you will see the filtered category IDs printed in the console or output area. These category IDs correspond to the categories specified in the `filterClasses` list, allowing you to work with specific categories of interest from the COCO dataset.

# In[17]:


# Get category ids that satisfy the given filter conditions
filterClasses = ['laptop', 'tv', 'cell phone']
# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
print(catIds)


# # **Step 8: Loading Category Information and Filtering Image IDs**
# 
# In this step, we load category information based on a specific category ID and filter image IDs that satisfy certain conditions. Follow these instructions to perform these tasks:
# 
# 1. Load category information: Set the variable `catID` to the specific category ID for which you want to load information. In this example, `catID` is set to `15`. Modify this value to match the category ID of interest.
# 
# 2. Load category information: Use the `loadCats(ids=catID)` method of the COCO API to load the category information corresponding to the provided category ID. This method returns a list of dictionaries containing details about the category, such as the category ID, name, and supercategory. In this example, we load the category information and print it using the line `print(coco.loadCats(ids=catID))`.
# 
# 3. Filter image IDs: Use the `getImgIds(catIds=[catID])` method of the COCO API to retrieve a list of image IDs that satisfy the given filter conditions. Pass the `catID` variable as an argument to the method within a list. This method returns a list of image IDs corresponding to the specified category ID. In this example, we fetch the image IDs and store the first ID in the variable `imgId` using the line `imgId = coco.getImgIds(catIds=[catID])[0]`.
# 
# 4. Print the image ID: Use the `print()` function to display the filtered image ID. In this example, we print the `imgId` variable, which contains the image ID that satisfies the given filter conditions.
# 
# Here is an example code snippet:
# 
# ```python
# # Load category information for the given ID
# catID = 15
# print(coco.loadCats(ids=catID))
# 
# # Get image ID that satisfies the given filter conditions
# imgId = coco.getImgIds(catIds=[catID])[0]
# print(imgId)
# ```
# 
# After executing this code, you will see the category information and the image ID printed in the console or output area. The category information includes details about the category with the provided ID, and the image ID corresponds to an image that belongs to that category in the COCO dataset.

# In[19]:


catID = 15
print(coco.loadCats(ids=catID))

# Get image ids that satisfy the given filter conditions
imgId = coco.getImgIds(catIds=[catID])[0]
print(imgId)


# # **9: Retrieving Annotation IDs for an Image**
# 
# In this step, we retrieve the annotation IDs for a specific image ID from the COCO dataset. Follow these instructions to accomplish this:
# 
# 1. Get the annotation IDs: Use the `getAnnIds(imgIds=[imgId], iscrowd=None)` method of the COCO API to retrieve the annotation IDs for the given image ID. Pass the `imgId` variable as an argument to the method within a list. Setting `iscrowd` to `None` ensures that both crowd and non-crowd annotations are included. This method returns a list of annotation IDs that correspond to the provided image ID. In this example, we fetch the annotation IDs and store them in the `ann_ids` variable using the line `ann_ids = coco.getAnnIds(imgIds=[imgId], iscrowd=None)`.
# 
# 2. Print the annotation IDs: Use the `print()` function to display the retrieved annotation IDs. In this example, we print the `ann_ids` variable, which contains the annotation IDs associated with the specified image.
# 
# 
# After executing this code, you will see the annotation IDs printed in the console or output area. These annotation IDs correspond to the annotations of objects present in the specified image ID in the COCO dataset.

# In[21]:


ann_ids = coco.getAnnIds(imgIds=[imgId], iscrowd=None)
print(ann_ids)


# # **Step 10: Displaying Image with Annotations**
# 
# In this step, we display an image from the COCO dataset along with its corresponding annotations. Follow these instructions to visualize the image and its annotations:
# 
# 1. Print image information: Use the `print()` function to display information about the image that will be visualized. In this example, we print the file name of the image using the line `print(image_path)`. This helps in identifying the image being displayed.
# 
# 2. Load and display the image: Use the `plt.imread()` function from the matplotlib library to load the image from the specified image directory (`imageDir + image_path`). Store the image in the `image` variable. Then, use the `plt.imshow()` function to display the image.
# 
# 3. Load and display the annotations: Use the `loadAnns()` method of the COCO API to load the annotations corresponding to the provided annotation IDs (`ann_ids`). Store the annotations in the `anns` variable. Then, use the `coco.showAnns()` method to display the annotations on top of the image. Set `draw_bbox=True` to draw bounding boxes around the annotated objects.
# 
# 4. Customize the plot: Use various functions from the `plt` module to customize the plot appearance. Use `plt.axis('off')` to turn off the axis labels, `plt.title()` to set a title for the plot, and `plt.tight_layout()` to optimize the layout of the plot.
# 
# 5. Display the plot: Finally, use `plt.show()` to display the plot with the image and annotations.
# 
# 
# After executing this code, a plot window will appear showing the image from the COCO dataset with the corresponding annotations displayed on it. The bounding boxes around the annotated objects will be visible, allowing you to visualize the annotations for the specified image ID.

# In[23]:


print(f"Annotations for Image ID {imgId}:")
anns = coco.loadAnns(ann_ids)

image_path = coco.loadImgs(imgId)[0]['file_name']
print(image_path)
image = plt.imread(imageDir + image_path)
plt.imshow(image)

# Display the specified annotations
coco.showAnns(anns, draw_bbox=True)

plt.axis('off')
plt.title('Annotations for Image ID: {}'.format(image_id))
plt.tight_layout()
plt.show()


# # **Step 11: Displaying Images with Annotations**
# 
# In this step, we will load images from the COCO dataset and display them along with their corresponding annotations. This will allow us to visualize the annotated objects in the images. Follow these instructions to accomplish this:
# 
# 1. Retrieve image IDs for specific categories: Use the `getImgIds()` method of the COCO API to retrieve a list of image IDs that contain specific categories. You can specify the category IDs or names using the `catIds` parameter. For example, `img_ids = coco.getImgIds(catIds=[1, 2, 3])` will retrieve image IDs that contain categories with IDs 1, 2, and 3.
# 
# 2. Select an image ID: Choose an image ID from the retrieved list to display. You can change the index value to select a different image. For example, `img_id = img_ids[0]` will select the first image ID from the list.
# 
# 3. Load image information: Use the `loadImgs()` method of the COCO API to load the image information for the selected image ID. This method returns a list of dictionaries containing details about the image, such as the file name, width, height, and URL. For example, `img_info = coco.loadImgs(img_id)[0]` will load the image information for the selected image ID.
# 
# 4. Retrieve annotation IDs: Use the `getAnnIds()` method of the COCO API to retrieve the annotation IDs for the selected image ID. This method returns a list of annotation IDs corresponding to the provided image ID. For example, `ann_ids = coco.getAnnIds(imgIds=[img_id])` will retrieve the annotation IDs for the selected image ID.
# 
# 5. Load annotations: Use the `loadAnns()` method of the COCO API to load the annotations corresponding to the retrieved annotation IDs. This method returns a list of annotation dictionaries containing information such as the category ID, bounding box coordinates, and segmentation mask. For example, `anns = coco.loadAnns(ann_ids)` will load the annotations for the selected image ID.
# 
# 6. Load and display the image: Use a suitable image processing library such as PIL or matplotlib to load and display the image. The image file path can be obtained from the image information dictionary. For example, `image = plt.imread(imageDir + img_info['file_name'])` will load the image using matplotlib.
# 
# 7. Display the annotations on the image: Use the `showAnns()` method of the COCO API to display the annotations on top of the image. This method takes the loaded annotations as input and optionally allows you to draw bounding boxes around the annotated objects. For example, `coco.showAnns(anns, draw_bbox=True)` will display the annotations with bounding boxes.
# 
# 8. Customize the plot: Use various functions from the chosen image processing library to customize the plot appearance. You can add titles, labels, or adjust the axis settings as desired.
# 
# 9. Show the plot: Finally, use the appropriate function from the image processing library to display the plot with the image and annotations. For example, `plt.show()` will display the plot using matplotlib.
# 
# 
# By following these steps, you will be able to load and display images from the COCO dataset along with their annotations, allowing you to visualize the annotated objects in the images.

# In[25]:


def main():

    # Category IDs.
    cat_ids = coco.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    print("Category ID -> Category Name:")
    print(
        f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    )

    # Category Name -> Category ID.
    query_name = cat_names[2]
    query_id = coco.getCatIds(catNms=[query_name])[0]
    print("Category Name -> ID:")
    print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco.getImgIds(catIds=[query_id])
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    # Pick one image.
    img_id = img_ids[2]
    img_info = coco.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    img_url = img_info["coco_url"]
    print(
        f"Image ID: {img_id}, File Name: {img_file_name}, Image URL: {img_url}"
    )

    # Get all the annotations for the specified image.
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    print(f"Annotations for Image ID {img_id}:")
    print(anns)

    # Use URL to load image.
    # im = Image.open(requests.get(img_url, stream=True).raw)
    # Load image from dataset
    im = plt.imread(imageDir+ coco.loadImgs(img_id)[0]['file_name'])
    # Save image and its labeled version.
    plt.axis("off")
    plt.imshow(np.asarray(im))
    plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
    # Plot segmentation and bounding box.
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)
    plt.show()
    return


if __name__ == "__main__":

    main()


# # **Step 12: Visualizing Category Distribution in the COCO Dataset**
# 
# In this step, we will visualize the distribution of categories in the COCO dataset using a horizontal bar plot. This will provide insights into the frequency of different object categories present in the dataset. Follow these instructions to create the plot:
# 
# 1. Load category information: Use the `getCatIds()` method of the COCO API to retrieve the category IDs present in the dataset. Store the category IDs in the `catIDs` variable. Then, use the `loadCats()` method to load the category information corresponding to the category IDs. Store the loaded categories in the `cats` variable.
# 
# 2. Get category names: Extract the category names from the loaded category information. Use a list comprehension to iterate over the `cats` variable and extract the `'name'` key from each category dictionary. Capitalize the category names using the `title()` method and store them in the `category_names` variable.
# 
# 3. Get category counts: Iterate over the category IDs in the `catIDs` variable and use the `getImgIds()` method to retrieve the image IDs associated with each category. Then, calculate the length of each image ID list to obtain the count of images for each category. Store the category counts in the `category_counts` variable.
# 
# 4. Create a color palette: Use the `sns.color_palette()` function from the seaborn library to create a color palette for the plot. Specify the desired color map ('viridis') and the number of colors based on the length of `category_names`. Store the colors in the `colors` variable.
# 
# 5. Create a horizontal bar plot: Create a figure with a specified size using `plt.figure(figsize=(11, 15))`. Use the `sns.barplot()` function from the seaborn library to create the horizontal bar plot. Pass the `category_counts` as the x-values, `category_names` as the y-values, and `colors` as the palette. This will create a bar for each category with its corresponding count.
# 
# 6. Add value labels to the bars: Iterate over the `category_counts` and `category_names` using `enumerate()`. Use the `plt.text()` function to add value labels to each bar. Specify the count as the text, `count + 20` as the x-coordinate for the label (to offset it from the bar), and `i` as the y-coordinate (to align it with the bar).
# 
# 7. Customize the plot: Add labels to the x-axis (`plt.xlabel()`) and y-axis (`plt.ylabel()`), and a title to the plot (`plt.title()`). Adjust the font sizes as desired. Use `plt.tight_layout()` to optimize the layout of the plot.
# 
# 8. Save and show the plot: Use `plt.savefig()` to save the plot as an image file (e.g., 'coco-cats.png') with a specified DPI (e.g., `dpi=300`). Finally, use `plt.show()` to display the plot.
# 
# After executing this code, a horizontal bar plot will be displayed showing the distribution of categories in the COCO dataset. Each category will be represented by a bar, with the count of images belonging to that category displayed on each bar. The plot provides an overview of the category distribution, helping to understand the composition of the dataset. The plot will also be saved as an image file named 'coco-cats.png' with a DPI of 300.

# In[27]:


# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

# Get category names
category_names = [cat['name'].title() for cat in cats]

# Get category counts
category_counts = [coco.getImgIds(catIds=[cat['id']]) for cat in cats]
category_counts = [len(img_ids) for img_ids in category_counts]


# Create a color palette for the plot
colors = sns.color_palette('viridis', len(category_names))

# Create a horizontal bar plot to visualize the category counts
plt.figure(figsize=(11, 15))
sns.barplot(x=category_counts, y=category_names, palette=colors)

# Add value labels to the bars
for i, count in enumerate(category_counts):
    plt.text(count + 20, i, str(count), va='center')
plt.xlabel('Count',fontsize=20)
plt.ylabel('Category',fontsize=20)
plt.title('Category Distribution in COCO Dataset',fontsize=25)
plt.tight_layout()
plt.savefig('coco-cats.png',dpi=300)
plt.show()


# # **Step 13: Visualizing Category Distribution as a Pie Chart**
# 
# In this step, we will visualize the distribution of categories in the COCO dataset as a pie chart. This type of chart allows for easy comparison of category proportions within the dataset. Follow these instructions to create the pie chart:
# 
# 1. Calculate category percentages: Compute the percentage of each category count out of the total count of all categories. Divide each category count by the total count and multiply by 100 to obtain the category percentages. Store the percentages in the `category_percentages` variable.
# 
# 2. Create the pie chart: Create a figure with a specified size using `plt.figure(figsize=(15, 24.9))`.
# 
# 3. Customize label properties: Define the labels for the pie chart using the category names and corresponding percentages. Add a space after each label for better readability. Customize the label properties, such as font size and background color, using the `label_props` dictionary.
# 
# 4. Add percentage information to labels: Use the `plt.pie()` function to create the pie chart. Pass the category counts as the data. Set `autopct=''` to hide the default percentage labels. Specify `startangle=90` to rotate the pie chart to start from the 90-degree angle (12 o'clock position). Use the `textprops` parameter to apply the label properties defined earlier. Set `pctdistance=0.85` to move the labels away from the center of the pie chart.
# 
# 5. Create the legend: Generate custom legend labels by combining the category labels and percentages. Use the `plt.legend()` function to create the legend. Pass the wedges (created in the previous step), the legend labels, and other parameters such as the title, location, and font size.
# 
# 6. Customize the plot: Adjust the plot aspect ratio using `plt.axis('equal')`. Add a title to the plot using `plt.title()`. Set the font size and adjust the layout of the plot using `plt.tight_layout()`.
# 
# 7. Save and show the plot: Use `plt.savefig()` to save the plot as an image file (e.g., 'coco-dis.png') with a specified DPI (e.g., `dpi=300`). Finally, use `plt.show()` to display the pie chart.
# 
# 
# After executing this code, a pie chart will be displayed showing the distribution of categories in the COCO dataset. Each category will be represented by a wedge in the pie, with the corresponding count and percentage displayed as labels. The legend will provide a clear overview of the category distribution, and the chart will be saved as an image file named 'coco-dis.png' with a DPI of 300.

# In[29]:


# Calculate percentage for each category
total_count = sum(category_counts)
category_percentages = [(count / total_count) * 100 for count in category_counts]


# Create a pie chart to visualize the category distribution
plt.figure(figsize=(15, 24.9))


# Customize labels properties
labels = [f"{name} " for name, percentage in zip(category_names, category_percentages)]
label_props = {"fontsize": 25, 
               "bbox": {"edgecolor": "white", 
                        "facecolor": "white", 
                        "alpha": 0.7, 
                        "pad": 0.5}
              }

# Add percentage information to labels, and set labeldistance to remove labels from the pie
wedges, _, autotexts = plt.pie(category_counts, 
                              autopct='', 
                              startangle=90, 
                              textprops=label_props, 
                              pctdistance=0.85)

# Create the legend with percentages
legend_labels = [f"{label}\n{category_percentages[i]:.1f}%" for i, label in enumerate(labels)]
plt.legend(wedges, legend_labels, title="Categories", loc="upper center", bbox_to_anchor=(0.5, -0.01), 
           ncol=4, fontsize=12)

plt.axis('equal')
plt.title('Category Distribution in COCO Dataset', fontsize=29)
plt.tight_layout()
plt.savefig('coco-dis.png', dpi=300)
plt.show()


# # **Step 14: Displaying Filtered Images with Annotations**
# 
# In this step, we will display images from the COCO dataset that contain specific classes, and visualize their annotations. Follow these instructions to accomplish this:
# 
# 1. Define the classes: Create a list called `filterClasses` containing the names of the classes you want to display. Only the images containing these classes will be shown.
# 
# 2. Fetch category IDs: Use the `getCatIds()` method of the COCO API to retrieve the category IDs corresponding to the `filterClasses`. Store the category IDs in the `catIds` variable.
# 
# 3. Get image IDs: Use the `getImgIds()` method of the COCO API to retrieve the image IDs that contain the desired category IDs. Pass the `catIds` variable as an argument to the method. Store the image IDs in the `imgIds` variable.
# 
# 4. Load a random image: Check if there are images in the `imgIds` list. If so, randomly select an image ID using `np.random.randint()`. Load the image information using the `loadImgs()` method and store it in the `image_info` variable.
# 
# 5. Load annotations: Get the annotation IDs for the selected image using the `getAnnIds()` method. Then, load the annotations using the `loadAnns()` method. Store the annotations in the `annotations` variable.
# 
# 6. Get category names and assign colors: Iterate over the annotations and use the `loadCats()` method to retrieve the category names based on the category IDs. Capitalize the category names and store them in the `category_names` variable. Assign colors to each category for visualization purposes.
# 
# 7. Load and display the image: Use `plt.imread()` to load the image using the `image_path` obtained from the image information. Display the image using `plt.imshow()`. Turn off the axis using `plt.axis('off')` and set the title of the plot using `plt.title()`. Save the plot as an image file if desired.
# 
# 8. Display bounding boxes and segmentations: Iterate over the annotations and use the bounding box and segmentation information to draw bounding boxes and segmented regions on the image. Use `patches.Rectangle()` to create bounding box rectangles and `plt.fill()` to display segmentation masks.
# 
# 9. Create a legend: Create a legend to associate the category names with their respective colors. Use `patches.Patch()` to create legend patches and `plt.legend()` to display the legend.
# 
# 10. Display the image with the legend: Show the image with the annotations and legend using `plt.show()`. Adjust the layout if needed.
# 
# 
# After executing this code, a random image containing the desired classes from the COCO dataset will be displayed with its annotations. The image will include bounding boxes around the objects and segmented regions indicated by different colors. Additionally, a legend will be shown to associate the colors with their respective category names. The plot can be saved as an image file named 'Img.png' (without annotations) and 'annImg.png' (with annotations) if desired.

# In[31]:


# Define the classes (out of the 80) which you want to see. Others will not be shown.
filterClasses = ['laptop', 'tv', 'cell phone']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)

# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)

# Load a random image from the filtered list
if len(imgIds) > 0:
    image_id = imgIds[np.random.randint(len(imgIds))]  # Select a random image ID
    image_info = coco.loadImgs(image_id)

    if image_info is not None and len(image_info) > 0:
        image_info = image_info[0]
        image_path = imageDir + image_info['file_name']

        # Load the annotations for the image
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        # Get category names and assign colors for annotations
        category_names = [coco.loadCats(ann['category_id'])[0]['name'].capitalize() for ann in annotations]
        category_colors = list(matplotlib.colors.TABLEAU_COLORS.values())

        # Load the image and plot it
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Annotations for Image ID: {}'.format(image_id))
        plt.tight_layout()
        plt.savefig('Img.png',dpi=350)
        plt.show()
        
        plt.imshow(image)
        plt.axis('off')

        # Display bounding boxes and segmented colors for each annotation
        for ann, color in zip(annotations, category_colors):
            bbox = ann['bbox']
            segmentation = ann['segmentation']

            # Display bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                                     edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

            # Display segmentation masks with assigned colors
            for seg in segmentation:
                poly = np.array(seg).reshape((len(seg) // 2, 2))
                plt.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.6)

        # Create a legend with category names and colors
        legend_patches = [patches.Patch(color=color, label=name) for color, name in zip(category_colors, category_names)]
        plt.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize='small')

        # Show the image with legend
        plt.title('Annotations for Image ID: {}'.format(image_id))
        plt.tight_layout()
        plt.savefig('annImg.png',dpi=350)
        plt.show()
    else:
        print("No image information found for the selected image ID.")
else:
    print("No images found for the desired classes.")


# ## **Step 15: Generating Masks for Object Segmentation**
# 
# In this step, we will explore the intricacies of generating masks specifically tailored for object segmentation tasks using the COCO dataset. We will delve into the different mask types provided by the dataset, discuss the significance of pixel-level mask annotations, and distinguish between binary and RGB masks. Additionally, we will delve into post-processing techniques, evaluation methods, and leveraging masks for fine-tuning deep learning models.
# 
# ### **Understanding Masks**
# 
# Masks play a crucial role in object segmentation tasks, providing valuable metadata to highlight specific regions or objects within an image. The COCO dataset offers three main types of masks:
# 
# 1. **Polygon Annotations**: Polygon annotations consist of a series of points connected by straight lines that enclose an object. They provide a set of coordinates representing the boundary of the object. However, using polygon annotations for object segmentation can be challenging due to potential ambiguity in defining accurate boundaries.
# 
# 2. **Instance Segmentation Masks**: Instance segmentation masks are binary or RGB masks that precisely indicate the boundaries separating different objects within an image. These masks provide highly accurate pixel-level segmentation for individual object instances. Instance segmentation masks are widely used in object segmentation tasks due to their accuracy and ability to label specific regions or objects within images.
# 
# 3. **Object Detection Bounding Boxes**: Object detection bounding boxes define rectangular regions that enclose objects within an image. They provide a higher-level representation of objects but lack the pixel-level detail offered by instance segmentation masks. Bounding boxes are less precise when objects overlap or have complex shapes.
# 
# ### **Pixel-Level Mask Annotations**
# 
# One crucial aspect of masks in the COCO dataset is the provision of pixel-level annotations. These annotations assign a binary value (0 or 1) to each pixel within the mask, enabling precise labeling of specific regions or objects within the image. This level of granularity enhances the accuracy and detail of object segmentation.
# 
# ### **Binary vs RGB Masks**
# 
# When generating masks for object segmentation tasks, it is important to understand the distinction between binary masks and RGB masks.
# 
# **Binary masks** are simpler and more efficient, consisting of black and white pixels. White pixels represent the object of interest, while black pixels represent the background. Binary masks are effective for identifying objects within the dataset but may lack nuanced color information.
# 
# **RGB masks**, on the other hand, provide a more detailed understanding of color information. They employ three color channels (red, green, and blue) with pixel values ranging from 0 to 255. Each pixel value represents a specific object instance within the image, allowing for more precise object separation and distinguishing between closely located or overlapping objects. However, generating RGB masks requires additional processing and can be computationally expensive.
# 
# ### **Generating and Utilizing Masks**
# 
# To generate masks from the COCO dataset, one can extract the segmented regions or masks using various techniques. This involves accessing the pixel-level information provided by the dataset or employing specialized tools to extract the masks accurately.
# 
# Furthermore, masks can be utilized in post-processing steps to refine their quality, such as applying morphological operations or smoothing techniques to enhance object boundaries and reduce noise.
# 
# Evaluating the accuracy of generated masks is paramount. Metrics such as Intersection over Union (IoU) can be used to measure the overlap between predicted masks and ground truth masks, providing insights into the quality of the segmentation.
# 
# Masks generated from the COCO dataset can also be employed to fine-tune deep learning models, enhancing their performance in object segmentation tasks. By utilizing masks as additional training data, models can learn to better discriminate between object classes and improve overall segmentation accuracy.
# 
# In conclusion, this step delves into the generation and utilization of masks for object segmentation using the COCO dataset. We explored different mask types, discussed the significance of pixel-level annotations, and highlighted the differences between binary and RGB masks. By comprehending the nuances of mask generation and leveraging them effectively, we can advance the accuracy and precision of object segmentation algorithms.
# 
# ## **Generating Masks for Object Segmentation**
# 
# In this section, we delve into the intricacies of generating masks specifically tailored for object segmentation using the COCO dataset. We present a systematic approach to extract mask information, generate binary masks, and employ post-processing techniques to enhance mask quality. Furthermore, we discuss evaluation methods to assess the accuracy of generated masks and the potential of leveraging masks for fine-tuning deep learning models.
# 
# ### **Extracting Mask Information**
# 
# To initiate the mask generation process, meticulous extraction of relevant mask information from the COCO dataset is imperative. Leveraging the COCO API, we load annotations associated with each image ID utilizing the `coco.loadAnns()` function. These annotations provide crucial details including object classes, segmentation polygons/masks, and bounding box coordinates.
# 
# ### **Generating Binary Masks**
# 
# Binary masks serve as a fundamental representation for object segmentation tasks. Utilizing pixel values of 0 or 1, binary masks effectively denote object absence or presence within the mask. To generate binary masks, we meticulously follow the following steps:
# 
# 1. Retrieve the image ID and corresponding annotations from the COCO dataset.
# 2. Iterate through the annotations, extracting segmentation polygons or masks.
# 3. Transform the segmentation polygons or masks into binary format, assigning a value of 1 to pixels within the object's boundary and 0 to background pixels.
# 4. Instantiate an empty binary mask with dimensions matching that of the image.
# 5. Overlay the binary mask onto the base image, thereby visually accentuating the accurately labeled object.
# 
# ### **Generating RGB Masks**
# 
# While binary masks provide a simplified representation, RGB masks offer a more nuanced understanding of object segmentation. RGB masks utilize three color channels (red, green, and blue) to differentiate between different classes or objects within an image. Generating RGB masks involves the following steps:
# 
# 1. Retrieve the image ID and corresponding annotations from the COCO dataset.
# 2. Iterate through the annotations, extracting segmentation polygons or masks.
# 3. Assign a unique color to each class or object within the image.
# 4. Instantiate an empty RGB mask with dimensions matching that of the image.
# 5. Overlay the RGB mask onto the base image, showcasing distinct colors for each segmented class or object.
# 
# RGB masks provide a detailed representation of object boundaries and facilitate more precise separation of objects, enabling finer-grained object segmentation.
# 
# ### **Generating Instance Segmentation Masks**
# 
# Instance segmentation masks offer a highly accurate representation of object boundaries and facilitate the separation of individual objects within an image. To generate instance segmentation masks, the following steps are followed:
# 
# 1. Retrieve the image ID and corresponding annotations from the COCO dataset.
# 2. Iterate through the annotations, extracting segmentation polygons or masks.
# 3. Generate a binary mask for each object instance, assigning a value of 1 to pixels within the object's boundary and 0 to background pixels.
# 4. Overlay the instance segmentation masks onto the base image, accurately delineating each individual object.
# 
# Instance segmentation masks provide a pixel-level segmentation of objects, enabling precise labeling and facilitating advanced object-based analyses.
# 
# ### **Generating Object Detection Bounding Boxes**
# 
# Object detection bounding boxes provide information about the rectangular regions encompassing objects within an image. While they do not offer pixel-level segmentation, they provide a coarse representation of object locations. To generate object detection bounding boxes, the following steps are executed:
# 
# 1. Retrieve the image ID and corresponding annotations from the COCO dataset.
# 2. Extract the bounding box coordinates for each object instance.
# 3. Overlay the bounding boxes onto the base image, visually highlighting the approximate object locations.
# 
# Object detection bounding boxes serve as a rudimentary means of identifying and localizing objects within an image, albeit with less precision compared to other mask types.
# 
# In conclusion, this section explored various mask generation techniques for object segmentation. We discussed the generation of binary masks, RGB masks, instance segmentation masks, and object detection bounding boxes. Each mask type offers distinct advantages and can be leveraged based on the specific requirements of the segmentation task at hand. By mastering these techniques, researchers can obtain accurate and detailed masks, enabling advanced object segmentation and analysis.
# 
# ### **Post-Processing Techniques**
# 
# Post-processing techniques play a pivotal role in refining mask quality and enhancing segmentation accuracy. Several key techniques warrant consideration:
# 
# 1. **Morphological Operations**: Employing morphological operations, such as erosion and dilation, facilitates boundary smoothing, gap filling, and the removal of small, isolated regions within masks.
# 2. **Smoothing Filters**: Leveraging smoothing filters, such as Gaussian blur, mitigates noise and augments the overall quality of the mask.
# 3. **Contour Detection**: Extraction of contours from binary masks provides precise object boundaries, thus enabling further analysis and evaluation.
# 
# The incorporation of these post-processing techniques can be realized through established libraries such as OpenCV or scikit-image.
# 
# ### **Evaluation of Generated Masks**
# 
# Accurate evaluation of generated masks is paramount to ensure robust object segmentation. Intersection over Union (IoU), a commonly employed evaluation metric, quantifies the overlap between predicted masks and ground truth masks, thereby furnishing insights into segmentation quality.
# 
# IoU calculation follows these steps:
# 
# 1. Compute the intersection area between the predicted mask and the ground truth mask.
# 2. Determine the union area by summing the areas of the predicted and ground truth masks, subsequently subtracting the intersection area.
# 3. Divide the intersection area by the union area to derive the IoU score.
# 
# ### **Leveraging Masks for Fine-tuning Models**
# 
# Masks derived from the COCO dataset provide a valuable resource for fine-tuning deep learning models, ultimately enhancing object segmentation performance. Integration of masks as supplementary training data empowers models to discern object classes more effectively and refine segmentation capabilities.
# 
# The fine-tuning process entails initializing a pre-trained model with weights and training it on a custom dataset incorporating the generated masks. This iterative training allows the model to adapt to the specific segmentation task, culminating in superior accuracy and refined object separation.
# 
# By harnessing masks for fine-tuning, models achieve enhanced accuracy, improved object localization, and superior generalization to unseen data.
# 
# In conclusion, this section elucidates the intricate art of mask generation for object segmentation using the esteemed COCO dataset. We delineate the process of extracting mask information, generating binary masks, employing post-processing techniques, evaluating mask accuracy, and leveraging masks for fine-tuning deep learning models. The profound insights garnered from this comprehensive approach empower researchers to make significant strides in the field of computer vision, transcending traditional boundaries and attaining remarkable results in object segmentation tasks.

# In[33]:


# Extracting Mask Information
# Load annotations for a specific image ID
# Load images for the given ids
image_ids = coco.getImgIds()
image_id = image_ids[0] 
annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))


# In[ ]:





# In[35]:


# Retrieve image file path
image_info = coco.loadImgs(image_id)[0]
image_dir = os.path.join(dataDir, 'images', 'val2014')
image_path = os.path.join(image_dir, image_info['file_name'])

# Load the main image
main_image = plt.imread(image_path)

# Create a new figure for displaying the main image
plt.figure(figsize=(10, 10))
plt.imshow(main_image)
plt.axis('off')
plt.title('Main Image')

# Save the figures
plt.savefig('main_image.png', dpi=300)

# Show the plots
plt.show()


# # Generating Binary Masks

# In[37]:


# Retrieve image dimensions
image_info = coco.loadImgs(image_id)[0]
height, width = image_info['height'], image_info['width']

# Create an empty binary mask with the same dimensions as the image
binary_mask = np.zeros((height, width), dtype=np.uint8)

# Iterate through the annotations and draw the binary masks
for annotation in annotations:
    segmentation = annotation['segmentation']
    mask = coco.annToMask(annotation)

    # Add the mask to the binary mask
    binary_mask += mask

# Display the binary mask
plt.figure(figsize=(10,10))
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')
plt.title('Binary Mask')
plt.savefig('binary_mask.png', dpi=300)
plt.show()


# # Generating RGB Mask

# In[39]:


# Retrieve image dimensions
image_info = coco.loadImgs(image_id)[0]
height, width = image_info['height'], image_info['width']

# Create an empty RGB mask with the same dimensions as the image
rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

# Define a color map for different object classes
color_map = {cat['id']: (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
             for cat in coco.loadCats(catIDs)}

# Iterate through the annotations and assign unique colors to each class/object
for annotation in annotations:
    category_id = annotation['category_id']
    color = color_map[category_id]

    # Draw the mask on the RGB mask
    mask = coco.annToMask(annotation)
    rgb_mask[mask == 1] = color

# Display the RGB mask
plt.figure(figsize=(10,10))
plt.imshow(rgb_mask)
plt.axis('off')
plt.title('RGB Mask')
plt.savefig('rgb_mask.png', dpi=300)
plt.show()


# # Generating Instance Segmentation Mask

# In[41]:


# Retrieve image dimensions
image_info = coco.loadImgs(image_id)[0]
height, width = image_info['height'], image_info['width']

# Create an empty mask with the same dimensions as the image
instance_mask = np.zeros((height, width), dtype=np.uint8)

# Iterate through the annotations and draw the instance segmentation masks
for annotation in annotations:
    segmentation = annotation['segmentation']
    mask = coco.annToMask(annotation)
    category_id = annotation['category_id']

    # Assign a unique value to each instance mask
    instance_mask[mask == 1] = category_id

# Display the instance segmentation mask
plt.figure(figsize=(10,10))
plt.imshow(instance_mask, cmap='viridis')
plt.axis('off')
plt.title('Instance Segmentation Mask')
plt.savefig('instance_mask.png', dpi=300)
plt.show()


# # Generating Object Detection Bounding Boxes
# 

# In[43]:


# Retrieve image dimensions
image_info = coco.loadImgs(image_id)[0]
height, width = image_info['height'], image_info['width']

# Create a new figure with the same dimensions as the image
fig, ax = plt.subplots(figsize=(10,10), dpi=100)

# Display the original image
ax.imshow(image)
ax.axis('off')
ax.set_title('Original Image')

# Draw bounding boxes on the original image
for annotation in annotations:
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    category_name = coco.loadCats(category_id)[0]['name']

    # Convert COCO bounding box format (x, y, width, height) to matplotlib format (xmin, ymin, xmax, ymax)
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    # Draw the bounding box rectangle
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add the category name as a label above the bounding box
    ax.text(xmin, ymin - 5, category_name, fontsize=8, color='red', weight='bold')

# Save the figure with adjusted dimensions
plt.savefig('bounding_boxes.png', bbox_inches='tight')

# Show the plot
plt.show()


# # Post-Processing Techniques
# 

# In[45]:


from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import gaussian

# Apply erosion to the binary mask
eroded_mask = binary_erosion(binary_mask)

# Apply dilation to the binary mask
dilated_mask = binary_dilation(binary_mask)

# Apply Gaussian blur to the binary mask
smoothed_mask = gaussian(binary_mask, sigma=2)

# Display the post-processed masks
fig, axes = plt.subplots(3,1, figsize=(12, 12))

axes[0].imshow(eroded_mask, cmap='gray')
axes[0].set_title('Eroded Mask')
axes[0].axis('off')

axes[1].imshow(dilated_mask, cmap='gray')
axes[1].set_title('Dilated Mask')
axes[1].axis('off')

axes[2].imshow(smoothed_mask, cmap='gray')
axes[2].set_title('Smoothed Mask')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('post_processed_masks.png', dpi=300)
plt.show()


# ## Evaluation of Generated Masks
# 
# Once we have generated masks for object segmentation tasks, it is essential to evaluate their quality and performance. Evaluation metrics provide insights into how well the generated masks align with the ground truth annotations. In this section, we will explore common evaluation metrics used in the context of generated masks.
# 
# ### Intersection over Union (IoU)
# Intersection over Union (IoU) is a widely used evaluation metric for measuring the similarity between masks. It calculates the ratio of the intersection area to the union area between the predicted mask and the ground truth mask. The IoU score ranges from 0 to 1, with a higher value indicating better alignment between the masks. A score of 1 indicates a perfect match, while a score of 0 suggests no overlap.
# 
# To calculate the IoU, we can use the numpy library to perform element-wise logical operations on the binary representations of the masks. The intersection and union areas are then computed, and the IoU score is derived by dividing the intersection by the union.
# 
# 

# In[47]:


import numpy as np

# Ground truth mask
gt_mask = binary_mask.astype(bool)  # Example ground truth mask

# Predicted mask
predicted_mask = smoothed_mask.astype(bool)  # Example predicted mask

# Calculate Intersection over Union (IoU)
intersection = np.logical_and(gt_mask, predicted_mask)
union = np.logical_or(gt_mask, predicted_mask)
iou = np.sum(intersection) / np.sum(union)

# Print the IoU score
print(f"Intersection over Union (IoU): {iou:.4f}")


# The Intersection over Union (IoU) score of 0.6959 indicates the degree of overlap between the predicted mask and the ground truth mask. A higher IoU score suggests a better alignment between the masks, indicating a closer resemblance to the ground truth annotations.
# 
# In this context, an IoU score of 0.6959 implies a moderate level of overlap and similarity between the predicted and ground truth masks. While it is not a perfect match, it indicates that the predicted mask captures a substantial portion of the objects outlined in the ground truth mask.
# 
# It is important to consider the specific requirements and objectives of the task when interpreting the IoU score. Depending on the application, a score of 0.6959 might be considered satisfactory or may require further improvement, which can be achieved through fine-tuning the model, adjusting parameters, or exploring alternative segmentation approaches.
# 
# Keep in mind that the IoU score is just one metric for evaluating the quality of generated masks. It is valuable to consider additional evaluation metrics, visual inspection, and qualitative assessment to obtain a comprehensive understanding of the segmentation performance.

# 
# 
# ### Other Evaluation Metrics
# In addition to IoU, there are several other evaluation metrics commonly used for mask evaluation. Some of these metrics include:
# 
# - Pixel Accuracy: Measures the percentage of correctly classified pixels in the predicted mask compared to the ground truth mask.
# - Precision and Recall: Evaluate the trade-off between true positives, false positives, and false negatives.
# - F1 Score: Combines precision and recall into a single metric to assess overall performance.
# - Mean Intersection over Union (mIoU): Computes the average IoU across multiple masks or classes.
# 
# The choice of evaluation metrics depends on the specific requirements of the task and the nature of the dataset. It is important to select the most appropriate metric(s) based on the objectives and characteristics of the segmentation problem.
# 
# ### Visualization and Qualitative Assessment
# Apart from numerical evaluation metrics, visual inspection and qualitative assessment of the generated masks are crucial for understanding the performance and identifying potential issues. Visualizing the masks overlaid on the corresponding images can provide insights into the accuracy of the segmentation and highlight areas that may require improvement.
# 
# Visualization techniques such as color-coded masks, bounding boxes, or contour overlays can help in visually comparing the predicted masks with the ground truth annotations. This visual assessment allows for a more comprehensive evaluation of the generated masks.
# 
# In summary, evaluating the quality of generated masks is essential to assess the performance of object segmentation algorithms. Utilizing appropriate evaluation metrics and conducting visual assessments can provide valuable insights for refining the models and improving the accuracy of the segmentation results.

# In[49]:


# Select an image ID for visualization
image_id = image_ids[0]

# Load the image
image_info = coco.loadImgs(image_id)[0]
image_path = os.path.join(imageDir, image_info['file_name'])
image = plt.imread(image_path)

# Get the ground truth annotations for the image
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)

# Create a blank image for overlaying the masks
overlay = image.copy()

# Iterate over the annotations and draw the masks on the overlay image
for annotation in annotations:
    # Get the segmentation mask
    mask = coco.annToMask(annotation)
    
    # Choose a random color for the mask
    color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    
    # Apply the mask to the overlay image
    overlay[mask == 1] = color

# Create a figure and subplot for visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original image
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

# Plot the image with overlay masks
ax2.imshow(overlay)
ax2.set_title('Masks Overlay')
ax2.axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Save the visualization as an image file
plt.savefig('mask_visualization.png', dpi=300)

# Show the plot
plt.show()


# # Step 16: Dataset Generation for Image and Mask
# 
# In this step, we will explore the process of generating the image and mask datasets using the COCO dataset. Dataset generation is a critical step in training deep learning models and performing various computer vision tasks, such as object segmentation and instance segmentation. 
# 
# ## Overview of Dataset Generation
# 
# The dataset generation process involves creating a dataset that consists of pairs of images and their corresponding masks. The image dataset contains the original images from the COCO dataset, while the mask dataset contains the binary or RGB masks associated with the objects within the images.
# 
# To generate the dataset, we need to follow a series of steps:
# 
# 1. **Data Preprocessing**: This initial step involves loading the COCO dataset using the COCO API and retrieving the necessary annotations and images. The annotations provide information about the objects, their categories, and their corresponding masks.
# 
# 2. **Image and Mask Pairing**: The next step is to pair each image from the COCO dataset with its corresponding mask. This ensures that each image and mask pair is aligned and can be used for training or evaluation purposes. The pairing process requires matching the image IDs with the corresponding mask IDs based on the object instances present in the annotations.
# 
# 3. **Image Augmentation**: To enhance the dataset's diversity and improve the model's generalization capabilities, image augmentation techniques can be applied. Image augmentation involves applying random transformations to the images, such as rotation, scaling, flipping, and cropping. These transformations create additional variations of the images and their corresponding masks, increasing the dataset size and introducing more diversity in the training data. It is crucial to ensure that the augmentation is applied consistently to both the images and the masks to maintain their alignment.
# 
# 4. **Splitting the Dataset**: Once the image and mask pairs are generated, it is common practice to split the dataset into training, validation, and test sets. The splitting ratio depends on factors such as the dataset size, the complexity of the task, and the availability of labeled data. The training set is used to train the model, the validation set is used to tune hyperparameters and monitor the model's performance, and the test set is used to evaluate the model's generalization on unseen data.
# 
# 5. **Dataset Storage**: Finally, the generated image and mask datasets need to be stored in a suitable format for easy retrieval during model training and evaluation. Common formats include TFRecord, HDF5, or simply storing the images and masks in separate directories. It is essential to organize the dataset structure and provide proper indexing or labeling to facilitate efficient data loading and management.
# 
# ## Function: Dataset Generator for COCO
# 
# To streamline the dataset generation process, we can create a versatile function called `dataset_generator_coco()` specifically designed for generating image and mask datasets from the COCO dataset. The function can be customized based on specific requirements and can take parameters such as `dataDir` (directory path to the COCO dataset), `dataType` (type of dataset: train, val, test), `catIds` (a list of category IDs to filter specific object classes), `imageAugmentation` (a flag indicating whether to apply image augmentation techniques), and `splitRatio` (the ratio for splitting the dataset).
# 
# The `dataset_generator_coco()` function performs the following steps:
# 
# 1. Initialize the COCO API and load the necessary annotations and images from the COCO dataset.
# 2. Filter the annotations based on the provided category IDs to focus on specific object classes if required.
# 3. Pair each image with its corresponding mask using the object instance information in the annotations.
# 4. Apply image augmentation techniques if `imageAugmentation` is set to `True`, ensuring consistent transformations are applied to both images and masks.
# 5. Split the dataset into training, validation, and test sets according to the specified `splitRatio`.
# 6. Store the generated image and mask datasets in a suitable format, such as TFRecord or HDF5, or organize them in separate directories.
# 
# The `dataset_generator_coco()` function provides a flexible and efficient approach to generate customized image and mask datasets from the COCO dataset. Researchers can utilize this function to generate datasets tailored to their specific computer vision tasks, ensuring the availability of properly aligned image and mask pairs.
# 
# It is important to adapt the function and its parameters to match the dataset structure and task requirements. Furthermore, it is recommended to maintain proper documentation and indexing to ensure easy access and management of the generated datasets throughout the training and evaluation process.

# In[12]:


import os
import numpy as np
import cv2
from skimage import io
import random

def dataset_generator_coco(dataDir, dataType, classes, splitRatio=(0.7, 0.15, 0.15),
                           input_image_size=(224, 224), batch_size=4, mode='train2014', mask_type='binary'):
    """
    Generate batches of images and masks from the COCO dataset.

    Args:
    - dataDir: Directory path where the COCO dataset is stored
    - dataType: Type of dataset (e.g., 'train2014', 'val2014', 'test2014')
    - classes: List of classes to include
    - splitRatio: Ratio for train-validation-test split (default: 0.7, 0.15, 0.15)
    - input_image_size: Size of the input image (default: (224, 224))
    - batch_size: Number of samples per batch (default: 4)
    - mode: Mode of the dataset (default: 'train2014')
    - mask_type: Type of mask to generate ('binary' or 'normal') (default: 'binary')

    Returns:
    - images: Batch of images
    - masks: Batch of masks
    """

    from pycocotools.coco import COCO

    # Step 1: Load COCO dataset
    coco = COCO(os.path.join(dataDir, f'annotations/instances_{dataType}.json'))

    img_folder = os.path.join(dataDir, 'images', mode)

    # Get image IDs for the specified category IDs
    catIds = coco.getCatIds(catNms=classes)
    imgIds = coco.getImgIds(catIds=catIds)

    # Step 2: Determine dataset size and number of batches
    dataset_size = len(imgIds)
    num_batches = int(np.ceil(dataset_size / batch_size))

    # Generate indices for shuffling the dataset
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # Iterate over each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]

        images = []
        masks = []

        # Process images and masks for the current batch
        for idx in batch_indices:
            # Get image and image object
            image_id = imgIds[idx]
            imageObj = coco.loadImgs(image_id)[0]

            # Read and normalize the image
            train_img = io.imread(os.path.join(img_folder, imageObj['file_name'])) / 255.0

            # Resize the image to the specified input size
            train_img = cv2.resize(train_img, input_image_size)

            # Check if the image is RGB or black and white
            if len(train_img.shape) == 3 and train_img.shape[2] == 3:
                # If RGB, no modification needed
                image = train_img
            else:
                # If black and white, increase dimensions to 3
                image = np.stack((train_img,) * 3, axis=-1)

            images.append(image)

            # Create the mask
            annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            train_mask = np.zeros(input_image_size)

            for a in range(len(anns)):
                # Get class name and pixel value for the mask
                className = coco.loadCats(anns[a]['category_id'])[0]['name']
                pixel_value = classes.index(className) + 1

                # Generate the mask and resize it to the specified input size
                mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)

                # Update the train_mask by taking the maximum value
                train_mask = np.maximum(mask, train_mask)

            # Add an extra dimension to match the train_img size [X * X * 3]
            train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
            masks.append(train_mask)

        yield np.array(images), np.array(masks)

# Usage example
dataDir = '/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/'
dataType = 'train2014'
classes = ['person']
batch_size = 4
dataset_size = 16
# Create a generator object for the COCO dataset
generator = dataset_generator_coco(dataDir, dataType, classes, batch_size=batch_size, mask_type='normal')

# Iterate over the generator to obtain image and mask batches
for i in range(dataset_size // batch_size):
    # Get the next batch of images and masks
    images, masks = next(generator)

    # Print batch information
    print(f'Batch {i + 1}:')
    print(f'Images shape: {images.shape}')
    print(f'Masks shape: {masks.shape}')
    print('')  # Add a blank line for better readability


# In[15]:


import matplotlib.pyplot as plt

def visualize_dataset(images, masks, num_samples=5):
    num_available_samples = images.shape[0]

    # Check if the number of requested samples is greater than the available samples
    if num_samples > num_available_samples:
        num_samples = num_available_samples

    # Randomly select a subset of samples
    indices = random.sample(range(num_available_samples), num_samples)

    # Iterate over the selected samples
    for idx in indices:
        image = images[idx]
        mask = masks[idx]

        # Plot the image and mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(mask[:, :, 0], cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()

# Usage example
dataDir = '/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/'
dataType = 'train2014'
classes = ['laptop']
batch_size = 4

generator = dataset_generator_coco(dataDir, dataType, classes, batch_size=batch_size, mask_type='normal')
images, masks = next(generator)
visualize_dataset(images, masks, num_samples=5)


# In[ ]:





# In[ ]:




