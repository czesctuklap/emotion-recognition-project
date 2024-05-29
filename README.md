# emotion-recognition-project
The project is based on the kNN algorithm, which classifies objects based on the class of their nearest neighbors. It involves calculating the distances between points, selecting the nearest \( k \) distances, and adopting the feature that dominates in the \( k \) set. We used the Euclidean distance for distance calculation, which for \( x_i \) and \( x_j \) in \( n \)-dimensional space is defined as:

\[d(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{i,k} − x_{j,k})^2}\]

where:

- \( x_i = (x_{i,1}, x_{i,2}, ..., x_{i,n}) \) is the feature vector for point \( i \),
- \( x_j = (x_{j,1}, x_{j,2}, ..., x_{j,n}) \) is the feature vector for point \( j \).

However, before the kNN algorithm is invoked, the data is normalized using the `StandardScaler()` function, labeled using the `LabelEncoder()`, and the dimensions are reduced using the `PCA` function.

Further in the code, the `preprocess-image()` function is used to process images before their use in the machine learning model. In this part, image normalization is also performed by subtracting the mean pixel value and dividing by the standard deviation:

\[ \text{img} = \frac{\text{img} - \text{mean(img)}}{\text{std(img)}} \]

where:

- `mean(img)` denotes the mean pixel value of the image,
- `std(img)` denotes the standard deviation of the pixel values of the image.

Next, the training data is augmented using the `ImageDataGenerator()` function, which applies various transformations to the images, such as rotations, shifts, zooms, and horizontal flips. Additionally, a function utilizing Histograms of Oriented Gradients (HOG) is defined, which extracts HOG features from both training and testing images. This sequence of steps prepares the data for machine learning, ensuring its diversity through augmentation and extracting important features using HOG, which can improve the model's ability to generalize and classification effectiveness.

As a result of the program's execution, we obtain a summary of actual features, predicted features, and visualization.

Dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
