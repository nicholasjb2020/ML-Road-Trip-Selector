# CS 4260 PA 4

### Developers: Nicholas Burns, Daniel Little

Running `main.py` will run the main script of our program, where you will be prompted for input.
- The user will input their preferences of each feature of a road trip on a scale of 1 - 10. The features are: 'Space', 'Wildlife', 'Festivals', 'History', 'Museums', 'Romantic', 'Nightlife', 'Architecture', 'Art', and 'Shopping'.
- A neural network (with three hidden layers and ReLU activation) will take the user preferences as input and output 1 of 338 clusters of road trips which are grouped based on similarities of features (same as those above).
- There are 1000 total road trips (which are each a set of 10 feature values and a cluster value), meaning there are about 3 road trips per cluster. You can view both the original dataset (which included a utility target metric that we deemed useless) and the clustered dataset in the data_files folder.
- The data was clustered using hierarchical clustering. This method yielded a higher silhouette score than k-means clustering and thus was selected to group the data. You can reproduce these tests using `hierarchical_test.py` and `k-means_test.py`.

If you wish to evaluate the model, uncomment line 131 in `train_network()` to do so.

Libraries needed: Numpy, Pandas, Scikit-Learn, Tensorflow, Keras
