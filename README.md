#UFOfinder.py

The function takes a directory of images and returns a csv of detections and images with horizon lines and detections overlaid.

For each image we:

##circle.py

###Identify blue scan circle, while being resilient to rainy images.

The function takes the first image and returns the centre and radius of the circle that captures the radar scan area.

* Find all pixels with RGB within bounds of 0,0,113.
* Draw the minimum bounding circle around those pixels using miniball package. This will also provide the central pixel.

##horizon.py

###Identify the angle of strongest horizon on the left and right sides of the circle.

The function takes an image and a scan circle and returns the angles of the left and right horizon, as well as a vector of mean RG/B at each angle.

* Set all all values outside the circle to zero.
* Iteratively take the mean RG/B of pixels intersecting each 1 degree line using the Bresenham's line algorithm.
* Take the angles of the max lines on the left and right side of the circle.
* Retain info on median RG/B at each degree.

##rain.py

###Determine whether an image should be excluded in the basis of bad weather.

The function takes an image, a scan circle, horizon angles, a horizon buffer angle and an RG/B threshold. It returns a boolean highlighting presence of weather anomalies.



##UFOs.py

###Find and characterise objects above the horizon.

The function takes a weather-free image, a scan circle, two horizon angles, a horizon buffer angle, a minimum distance, a target intensity, a lower intensity interval, an upper intensity interval, a smoothing matrix, and an output directory. It returns a table of object detections, including  x, y, angle, distance, and intensity.

##flag\_radial_artifacts.py

###Flag detections that should be regarded as radial artifacts.

The function takes a table of detections, a table of RG/B at each degree, and an RG/B median threshold. It adds a column to the table of detections highlighting whether each detection is in a radial artifact zone.