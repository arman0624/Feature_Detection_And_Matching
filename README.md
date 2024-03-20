The goal of feature detection and matching is to identify a pairing between a point in one image and a corresponding point in another image. These correspondences can then be used to stitch multiple images together into a panorama.

This program detects discriminative features (which are reasonably invariant to translation, rotation, and illumination) in an image and find the best matching features in another image.

To help visualize the results and debug the program, there is user interface that displays detected features and best matches in another image. There is also an ORB feature detector, a popular technique in the vision community, for comparison.

Using the UI and code, you can load in a set of images, view the detected features, and visualize the feature matches that your algorithm computes. 

By running featuresUI.py, you will see a UI where you have the following choices:

    Keypoint Detection
    You can load an image and compute the points of interest with their orientation.
    Feature Matching
    Here you can load two images and view the computed best matches using the specified algorithms.
    Benchmark
    After specifying the path for the directory containing the dataset, the program will run the specified algorithms on all images and compute ROC curves for each image.

The UI is a tool to help you visualize the keypoint detections and feature matching results.

You can go out and take some photos of your own to see how well this program works on more interesting data sets. For example, you could take images of a few different objects (e.g., books, offices, buildings, etc.) and see how well it works.