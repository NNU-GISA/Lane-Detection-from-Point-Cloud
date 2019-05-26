# Final project 5
# Object Detection in Point Cloud: Lane marking

* **Authors:** Zunran Guo, Feiyu Chen  
* **Time:** June 5th, 2019  
* **Course:**  ELEC_ENG 395, 495: Geospatial Vision and Visualization 

* **Report:** FinalProject_Feiyu_Zunran.pdf  
Or view it on this [google slide](https://docs.google.com/presentation/d/1qCYxsXetAjxamBPe9fV-3Cld6QCXfZCR1QGav4J_9lM/edit?usp=sharing).


# Task
Find lanes from a point cloud data.

# Main file
> main.ipynb  

Run through it using jupyter notebook to obtain the lane detection result.

# Data

* File:  
[data/final_project_point_cloud.fuse](data/final_project_point_cloud.fuse)


* Format:  
Each row uses 4 number to represent a point:
	```
	1. latitude (degree)
	2. longitude (degree)
	3. altitude (meter)
	4. point reflexion intensity (0~100)
	```


# Dependencies
Please use **Python3** and **jupyter notebook** to run the program.  
The command for install the required packages is:  
> $ pip install jupyter numpy scipy sklearn matplotlib open3d-python opencv-python    

Result will be displayed inside the jupyter notebook, as well as saved to the "result/" folder. You could use the "pcl_viewer" to view the saved ".pcd" files. 

To install "pcl_viewer":  
> $ sudo apt-get install pcl-tool