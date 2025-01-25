# -*- coding: utf-8 -*-
import numpy as np
from spectral import *
from scipy import *
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



from spectral import envi

# Open hyperspectral data file
#pure stevia

def calibrate_image(raw_image, white_ref, dark_ref):
    # Load image and references
    raw_data = raw_image.load()
    white_data = white_ref.load()
    dark_data = dark_ref.load()

    # Subtract dark reference from raw image and white reference
    calibrated_data = raw_data - dark_data
    normalized_white = white_data - dark_data

    # Avoid division by zero
    normalized_white[normalized_white == 0] = 1e-10

    # Divide calibrated data by normalized white reference
    calibrated_data /= normalized_white

    return calibrated_data


# Open hyperspectral data file for pure stevia
stevia = envi.open(
    r'C:\hyspec imgs\all images\pure stevia\capture\pure stevia.hdr',
    r'C:\hyspec imgs\all images\pure stevia\capture\pure stevia.raw'
)

# Open white and dark reference files for pure stevia
white_reference_stevia = envi.open(
    r'C:\hyspec imgs\all images\pure stevia\capture\WHITEREF_pure stevia.hdr'
)
dark_reference_stevia = envi.open(
    r'C:\hyspec imgs\all images\pure stevia\capture\DARKREF_pure stevia.hdr'
)
calibrated_stevia = calibrate_image(stevia, white_reference_stevia, dark_reference_stevia)
# Open hyperspectral data file for pure maltodextrin
pure_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\pure maltodextrin\capture\pure maltodextrin .hdr',
    r'C:/hyspec imgs/all images/all images zip_1/pure maltodextrin/capture/pure maltodextrin .raw'
)
white_reference_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\pure maltodextrin\capture\WHITEREF_pure maltodextrin .hdr'
)
dark_reference_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\pure maltodextrin\capture\DARKREF_pure maltodextrin .hdr'
)
calibrated_maltodextrin = calibrate_image(pure_maltodextrin, white_reference_maltodextrin, dark_reference_maltodextrin)

# Use the same white and dark reference files for maltodextrin as needed


# Open hyperspectral data file for pure saccharin
pure_saccharin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\saccharin - 1\capture\saccharin - 1.hdr',
    r'C:\hyspec imgs\all images\all images zip_1\saccharin - 1\capture\saccharin - 1.raw'
)

# Open white and dark reference files for saccharin
white_reference_saccharin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\saccharin - 1\capture\WHITEREF_saccharin - 1.hdr'
    
)
dark_reference_saccharin = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\saccharin - 1\capture\DARKREF_saccharin - 1.hdr'
)
calibrated_saccharin = calibrate_image(pure_saccharin, white_reference_saccharin, dark_reference_saccharin)


# Open hyperspectral data file for pure erythritol
pure_erythritol = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\pure erythritol\capture\pure erythritol.hdr',
    r'C:\hyspec imgs\all images\all images zip_1\pure erythritol\capture\pure erythritol.raw'
)

# Open white and dark reference files for erythritol
white_reference_erythritol = envi.open(
    r'C:\hyspec imgs\all images\all images zip_1\pure erythritol\capture\WHITEREF_pure erythritol.hdr'
)
dark_reference_erythritol = envi.open(
    r'C:/hyspec imgs/all images/all images zip_1/pure erythritol/capture/DARKREF_pure erythritol.hdr'
)
calibrated_erythritol = calibrate_image(pure_erythritol, white_reference_erythritol, dark_reference_erythritol)

# Open hyperspectral data file for 5% saccharin mixture
five_percent_saccharin = envi.open(
    r'C:\hyspec imgs\all images\5% mixture\capture\5% mixture.hdr',
    r'C:\hyspec imgs\all images\5% mixture\capture\5% mixture.raw'
)

# Open white and dark reference files for 5% saccharin mixture
white_reference_5_saccharin = envi.open(
    r'C:\hyspec imgs\all images\5% mixture\capture\WHITEREF_5% mixture.hdr'
)
dark_reference_5_saccharin = envi.open(
    r'C:\hyspec imgs\all images\5% mixture\capture\DARKREF_5% mixture.hdr'
)
calibrated_5_saccharin = calibrate_image(five_percent_saccharin, white_reference_5_saccharin, dark_reference_5_saccharin)


# Open hyperspectral data file for 15% saccharin mixture
fifteen_percent_saccharin = envi.open(
    r'C:\hyspec imgs\all images\15% mixture\capture\15% mixture.hdr',
    r'C:\hyspec imgs\all images\15% mixture\capture\15% mixture.raw'
)

# Open white and dark reference files for 15% saccharin mixture
white_reference_15_saccharin = envi.open(
    r'C:\hyspec imgs\all images\15% mixture\capture\WHITEREF_15% mixture.hdr'
)
dark_reference_15_saccharin = envi.open(
    r'C:\hyspec imgs\all images\15% mixture\capture\DARKREF_15% mixture.hdr'
)
calibrated_15_saccharin = calibrate_image(fifteen_percent_saccharin, white_reference_15_saccharin, dark_reference_15_saccharin)


# Open hyperspectral data file for 5% erythritol mixture
five_percent_erythritol = envi.open(
    r'C:\hyspec imgs\all images\5% erythritol mixture\capture\5% erythritol mixture.hdr',
    r'C:\hyspec imgs\all images\5% erythritol mixture\capture\5% erythritol mixture.raw'
)

# Open white and dark reference files for 5% erythritol mixture
white_reference_5_erythritol = envi.open(
    r'C:\hyspec imgs\all images\5% erythritol mixture\capture\WHITEREF_5% erythritol mixture.hdr'
)
dark_reference_5_erythritol = envi.open(
    r'C:\hyspec imgs\all images\5% erythritol mixture\capture\DARKREF_5% erythritol mixture.hdr'
)
calibrated_5_erythritol = calibrate_image(five_percent_erythritol, white_reference_5_erythritol, dark_reference_5_erythritol)


# Open hyperspectral data file for 15% erythritol mixture
fifteen_percent_erythritol = envi.open(
    r'C:\hyspec imgs\all images\15% erythritol mixture\capture\15% erythritol mixture.hdr',
    r'C:/hyspec imgs/all images/15% erythritol mixture/capture/15% erythritol mixture.raw'
    )

# Open white and dark reference files for 15% erythritol mixture
white_reference_15_erythritol = envi.open(
    r'C:/hyspec imgs/all images/15% erythritol mixture/capture/WHITEREF_15% erythritol mixture.hdr'
)
dark_reference_15_erythritol = envi.open(
    r'C:/hyspec imgs/all images/15% erythritol mixture/capture/DARKREF_15% erythritol mixture.hdr'
)
calibrated_15_erythritol = calibrate_image(fifteen_percent_erythritol, white_reference_15_erythritol, dark_reference_15_erythritol)

# Open hyperspectral data file for 5% maltodextrin mixture
five_percent_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\5% maltodextrin mixture\capture\5% maltodextrin mixture.hdr',
    r'C:\hyspec imgs\all images\5% maltodextrin mixture\capture\5% maltodextrin mixture.raw'
    )

# Open white and dark reference files for 5% maltodextrin mixture
white_reference_5_maltodextrin = envi.open(
    r'C:/hyspec imgs/all images/5% maltodextrin mixture/capture/WHITEREF_5% maltodextrin mixture.hdr'
)
dark_reference_5_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\5% maltodextrin mixture\capture\DARKREF_5% maltodextrin mixture.hdr'
    )
calibrated_5_maltodextrin = calibrate_image(five_percent_maltodextrin, white_reference_5_maltodextrin, dark_reference_5_maltodextrin)

# Open hyperspectral data file for 15% maltodextrin mixture
fifteen_percent_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\15% maltodextrin mixture\capture\15% maltodextrin mixture.hdr',
    r'C:/hyspec imgs/all images/15% maltodextrin mixture/capture/15% maltodextrin mixture.raw'
)

# Open white and dark reference files for 15% maltodextrin mixture
white_reference_15_maltodextrin = envi.open(
    r'C:\hyspec imgs\all images\15% maltodextrin mixture\capture\WHITEREF_15% maltodextrin mixture.hdr'
)
dark_reference_15_maltodextrin = envi.open(
    r'C:/hyspec imgs/all images/15% maltodextrin mixture/capture/DARKREF_15% maltodextrin mixture.hdr'
)
calibrated_15_maltodextrin = calibrate_image(fifteen_percent_maltodextrin, white_reference_15_maltodextrin, dark_reference_15_maltodextrin)

#dispplay caliberated image
wmp1 = calibrated_stevia

#wmp = np.rot90(wmp, k=1, axes=(1, 0))
trimmed_wmp_purestevia = wmp1[200:300, 150:300, :]

wmp2 = calibrated_erythritol
trimmed_wmp_pure_erythyritol= wmp2[200:300, 170:270, :]

wmp3 = calibrated_maltodextrin
trimmed_wmp_pure_maltodextrin= wmp3[220:300, 150:280, :]


wmp4 = calibrated_saccharin
trimmed_wmp_pure_saccharin= wmp4[200:300, 175:270, :]

wmp5 = calibrated_15_saccharin
trimmed_wmp_15_saccharin= wmp5[210:300, 190:280, :]

wmp6 = calibrated_5_saccharin
trimmed_wmp_5_saccharin= wmp6[210:300, 170:280, :]

wmp7 = calibrated_5_erythritol
trimmed_wmp_5_erythritol= wmp7[210:300, 180:280, :]

wmp8 = calibrated_15_erythritol
trimmed_wmp_15_erythritol= wmp8[210:300, 180:280, :]

wmp9 = calibrated_5_maltodextrin
trimmed_wmp_5_maltodextrin= wmp9[210:300, 160:280, :]

wmp10 = calibrated_15_maltodextrin
trimmed_wmp_15_maltodextrin= wmp10[200:310, 170:270, :]

from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# Function to apply SavGol filter and Min-Max Scaling to a hyperspectral image
def preprocess_image(sample):
    # Apply Savitzky-Golay filter across each band (axis 2 is the spectral dimension)
    filtered_sample = savgol_filter(sample, window_length=11, polyorder=3, axis=2)
    
    # Flatten spatial dimensions for scaling, then apply Min-Max Scaling
    reshaped_sample = filtered_sample.reshape(-1, filtered_sample.shape[2])
    scaler = MinMaxScaler()
    scaled_sample = scaler.fit_transform(reshaped_sample)
    
    # Reshape back to original spatial dimensions with scaled spectral bands
    scaled_sample = scaled_sample.reshape(filtered_sample.shape)
    return scaled_sample

# Apply the preprocessing to each trimmed hyperspectral image
processed_stevia = preprocess_image(trimmed_wmp_purestevia)
processed_erythritol = preprocess_image(trimmed_wmp_pure_erythyritol)
processed_maltodextrin = preprocess_image(trimmed_wmp_pure_maltodextrin)
processed_saccharin = preprocess_image(trimmed_wmp_pure_saccharin)
processed_15_saccharin = preprocess_image(trimmed_wmp_15_saccharin)
processed_5_saccharin = preprocess_image(trimmed_wmp_5_saccharin)
processed_5_erythritol = preprocess_image(trimmed_wmp_5_erythritol)
processed_15_erythritol = preprocess_image(trimmed_wmp_15_erythritol)
processed_5_maltodextrin = preprocess_image(trimmed_wmp_5_maltodextrin)
processed_15_maltodextrin = preprocess_image(trimmed_wmp_15_maltodextrin)

# Optionally, display one of the processed images as an example
plt.imshow(processed_stevia[:, :, 50], cmap='viridis')  # Display a specific band
plt.colorbar()
plt.title("Processed Stevia Sample - Band 50")
plt.show()

#PCA WITH linear kernel
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply PCA to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM model with a linear kernel on PCA-transformed data
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = svm_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Linear SVM with PCA')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))
# Hyperspectral image data and labels
data = {
    'pure stevia': processed_stevia,
    
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    
    
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}


#rbf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply PCA to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM model on PCA-transformed data with RBF kernel
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = svm_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (RBF with PCA)')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize PCA-reduced data in 2D for clarity (using first two components)
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label, alpha=0.5)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA-Reduced Data Visualization')
plt.legend()
plt.show()

#logistic regression PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure erythritol': processed_erythritol,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply PCA to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Multinomial Logistic Regression model on PCA-transformed data
log_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = log_reg_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Logistic Regression with PCA)')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize PCA-reduced data in 2D for clarity (using first two components)
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label, alpha=0.5)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA-Reduced Data Visualization')
plt.legend()
plt.show()


#random forest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply PCA to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Random Forest model on PCA-transformed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_model.predict(X_test)

# Display confusion matrix with a pink/purple color map
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples',  # Change color to purple shades
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (RF with PCA')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize PCA-reduced data in 2D for clarity (using first two components)
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label, alpha=0.5)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA-Reduced Data Visualization')
plt.legend()
plt.show()

#factor analysis with random forest
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply Factor Analysis to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
factor_analysis = FactorAnalysis(n_components=n_components, random_state=42)
X_fa = factor_analysis.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_fa, y, test_size=0.2, random_state=42)

# Train Random Forest model on Factor Analysis-transformed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_model.predict(X_test)

# Display confusion matrix with a yellow color map
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr',  # Change color to yellow tones
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (RF with FA')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize Factor Analysis-reduced data in 2D for clarity (using first two components)
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(X_fa[y == label, 0], X_fa[y == label, 1], label=label, alpha=0.5)

plt.xlabel('Factor Analysis Component 1')
plt.ylabel('Factor Analysis Component 2')
plt.title('Factor Analysis-Reduced Data Visualization')
plt.legend()
plt.show()

#factor analysis with moultimial reg
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Helper function to augment images by rotating them at angles 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    h, w, bands = image.shape
    augmented_images = augment_image(image)
    for aug_image in augmented_images:
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))  # Assign label for each pixel in the image

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply Factor Analysis to reduce dimensionality
n_components = 5  # Number of components to keep (adjustable)
factor_analysis = FactorAnalysis(n_components=n_components, random_state=42)
X_fa = factor_analysis.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_fa, y, test_size=0.2, random_state=42)

# Train Multinomial Logistic Regression model on Factor Analysis-transformed data
logreg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
logreg_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = logreg_model.predict(X_test)

# Display confusion matrix with an orange color map
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',  # Change color to orange tones
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (FA with Multinomial Reg')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Visualize Factor Analysis-reduced data in 2D for clarity (using first two components)
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(X_fa[y == label, 0], X_fa[y == label, 1], label=label, alpha=0.5)

plt.xlabel('Factor Analysis Component 1')
plt.ylabel('Factor Analysis Component 2')
plt.title('Factor Analysis-Reduced Data Visualization')
plt.legend()
plt.show()

#wavelet transfrom with svm
import numpy as np
import pywt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary (Assumed to be loaded as per your initial structure)
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Define augmentation angles
augmentation_angles = [0, 90, 180]  # Rotation angles

# Function to rotate and augment data
def augment_data(data, angles):
    augmented_X, augmented_y = [], []
    for label, image in data.items():
        h, w, bands = image.shape
        for angle in angles:
            # Rotate each band separately to maintain spectral integrity
            rotated_image = np.stack([np.rot90(image[:, :, band], k=angle // 90) for band in range(bands)], axis=-1)
            augmented_X.append(rotated_image.reshape(-1, bands))  # Flatten spatial dimensions
            augmented_y.extend([label] * (h * w))  # Assign label for each pixel
    return np.vstack(augmented_X), np.array(augmented_y)

# Augment data
X_augmented, y_augmented = augment_data(data, augmentation_angles)

# Wavelet transform function
def apply_wavelet_transform(X, wavelet='haar'):
    transformed_features = []
    for sample in X:
        transformed_sample = []
        for band_index in range(sample.shape[-1]):  # Loop through bands
            band = sample[:, band_index]  # Get the individual band data (1D per pixel)
            coeffs = pywt.dwt(band, wavelet)  # Apply wavelet transform on 1D band data
            cA, _ = coeffs  # Approximation coefficients only
            transformed_sample.extend(cA)  # Store approximation coefficients
        transformed_features.append(transformed_sample)
    return np.array(transformed_features)

# Apply wavelet transform
X_wavelet = apply_wavelet_transform(X_augmented)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_wavelet, y_augmented, test_size=0.2, random_state=42)

# Train SVM with RBF kernel on wavelet-transformed data
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM with Wavelet Transform')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))


import numpy as np
import pywt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary (Assumed to be loaded as per your initial structure)
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Define augmentation angles
augmentation_angles = [0, 90, 180]  # 4 rotation angles

# Function to rotate and augment data
def augment_data(data, angles):
    augmented_X, augmented_y = [], []
    for label, image in data.items():
        h, w, bands = image.shape
        for angle in angles:
            # Rotate each band separately to maintain spectral integrity
            rotated_image = np.stack([np.rot90(image[:, :, band], k=angle // 90) for band in range(bands)], axis=-1)
            augmented_X.append(rotated_image.reshape(-1, bands))  # Flatten spatial dimensions
            augmented_y.extend([label] * (h * w))  # Assign label for each pixel
    return np.vstack(augmented_X), np.array(augmented_y)

# Augment data
X_augmented, y_augmented = augment_data(data, augmentation_angles)

# Wavelet transform function
def apply_wavelet_transform(X, wavelet='haar'):
    transformed_features = []
    for sample in X:
        transformed_sample = []
        for band_index in range(sample.shape[-1]):  # Loop through bands
            band = sample[band_index]  # Get the individual band data
            coeffs = pywt.dwt(band, wavelet)  # Apply wavelet transform on 1D band data
            cA, _ = coeffs  # Approximation coefficients only
            transformed_sample.extend(cA)  # Store approximation coefficients
        transformed_features.append(transformed_sample)
    return np.array(transformed_features)

# Apply wavelet transform
X_wavelet = apply_wavelet_transform(X_augmented)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_wavelet, y_augmented, test_size=0.2, random_state=42)

# Train SVM with RBF kernel on wavelet-transformed data
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM with Wavelet Transform')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

#lda with svm
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Original data dictionary (assumed to be loaded in the same structure as before)
data = {
    'pure stevia': processed_stevia,
    'pure erythritol': processed_erythritol,
    '5% erythritol': processed_5_erythritol,
    '15% erythritol': processed_15_erythritol,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Function to augment data by rotating images
def augment_data(image):
    augmented_images = [image]  # Start with the original image
    for angle in [0, 90, 180]:  # Rotate the image at specified angles
        rotated_image = rotate(image, angle, reshape=False)  # Rotate without reshaping
        augmented_images.append(rotated_image)
    return augmented_images

# Flatten each image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_data(image)
    for aug_image in augmented_images:
        h, w, bands = aug_image.shape
        X.append(aug_image.reshape(h * w, bands))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                 # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spectral_bands)
y = np.array(y)      # Shape: (total_samples,)

# Apply LDA to reduce dimensionality
n_components = min(len(data) - 1, X.shape[1])  # LDA max components = classes - 1
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Train SVM with RBF kernel on LDA-transformed data
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='brown',  # Change cmap to 'brown'
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM with LDA')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))



#lda with augment random
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import rotate

# Original data dictionary (assumed to be loaded in the same structure as before)
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Flatten each image and create labels with augmentation
X = []
y = []
angles = [0, 90, 180]  # Using rotation angles for data augmentation

for label, image in data.items():
    h, w, bands = image.shape
    for angle in angles:
        rotated_image = rotate(image, angle, resize=False)
        X.append(rotated_image.reshape(h * w, bands))
        y.extend([label] * (h * w))

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Apply LDA to reduce dimensionality
lda = LinearDiscriminantAnalysis(n_components=len(data) - 1)
X_lda = lda.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Train Random Forest on LDA-transformed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

# Plotting confusion matrix with violet color map
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest with LDA')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# Cross-validation accuracy
scores = cross_val_score(rf_model, X_lda, y, cv=5)
print("Cross-Validation Accuracy Scores:", scores)
print("Mean Accuracy:", np.mean(scores))

# Optional: Visualize data in 2D (for illustrative purposes, using PCA or another dimensionality reduction technique if needed)


#morphological




#morpho with random

import numpy as np
from skimage.morphology import opening, closing, disk
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Morphological Profile function for spatial feature extraction
def morphological_profile(image, selem=disk(1)):
    proimport numpy as np
from skimage.morphology import opening, closing, disk
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary without erythritol or its mixtures
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Morphological Profile function for spatial feature extraction
def morphological_profile(image, selem=disk(1)):
    profile_features = []
    h, w, bands = image.shape
    
    for band in range(bands):
        band_image = image[:, :, band]
        
        # Apply opening and closing
        opened = opening(band_image, selem)
        closed = closing(band_image, selem)
        
        # Append the results to profile features
        profile_features.append(opened)
        profile_features.append(closed)
    
    return np.stack(profile_features, axis=2)

# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, resize=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        mp_image = morphological_profile(aug_image)  # Extract spatial features
        h, w, features = mp_image.shape
        X.append(mp_image.reshape(h * w, features))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                  # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spatial_features)
y = np.array(y)      # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=list(data.keys())))
    
    # Plot heatmap of confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
                xticklabels=list(data.keys()), yticklabels=list(data.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name} with Morphological Profiles')
    plt.show()
    
import numpy as np
from skimage.morphology import opening, closing, disk
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary without erythritol or its mixtures
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Morphological Profile function for spatial feature extraction
def morphological_profile(image, selem=disk(1)):
    profile_features = []
    h, w, bands = image.shape
    
    for band in range(bands):
        band_image = image[:, :, band]
        
        # Apply opening and closing
        opened = opening(band_image, selem)
        closed = closing(band_image, selem)
        
        # Append the results to profile features
        profile_features.append(opened)
        profile_features.append(closed)
    
    return np.stack(profile_features, axis=2)

# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, resize=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        mp_image = morphological_profile(aug_image)  # Extract spatial features
        h, w, features = mp_image.shape
        X.append(mp_image.reshape(h * w, features))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                  # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spatial_features)
y = np.array(y)      # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=list(data.keys())))
    
    # Plot heatmap of confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
                xticklabels=list(data.keys()), yticklabels=list(data.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name} with Morphological Profiles')
    plt.show()
file_features = []
    h, w, bands = image.shape
    
    for band in range(bands):
        band_image = image[:, :, band]
        
        # Apply opening and closing
        opened = opening(band_image, selem)
        closed = closing(band_image, selem)
        
        # Append the results to profile features
        profile_features.append(opened)
        profile_features.append(closed)
    
    return np.stack(profile_features, axis=2)

# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        mp_image = morphological_profile(aug_image)  # Extract spatial features
        h, w, features = mp_image.shape
        X.append(mp_image.reshape(h * w, features))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                  # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spatial_features)
y = np.array(y)      # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model on morphological profile-transformed data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_model.predict(X_test)

# Display confusion matrix with pink color scheme
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='pink', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest with Morphological Profiles')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))


#morpho with multinomial

import numpy as np
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, axes=(0, 1), reshape=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        h, w, c = aug_image.shape
        X.append(aug_image.reshape(h * w, c))  # Flatten spatial dimensions
        y.extend([label] * (h * w))             # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)  # Shape: (total_samples, spectral_features)
y = np.array(y)   # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_model = SVC(kernel='linear', random_state=42)

# Perform Recursive Feature Elimination (RFE)
selector = RFE(estimator=svm_model, n_features_to_select=10)  # Change n_features_to_select as needed
selector = selector.fit(X_train, y_train)

# Reduce X_train and X_test based on selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train SVM model on the selected features
svm_model.fit(X_train_selected, y_train)

# Make predictions on test data
y_pred = svm_model.predict(X_test_selected)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys', 
            xticklabels=list(data.keys()), yticklabels=list(data.keys()))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - SVM with RFE')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))


#Recursive Feature Elimination and SVM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for SVM processing
X = []
y = []

# Assuming these are your pre-processed images; replace with your actual variable names
for label, img in enumerate([processed_stevia, processed_erythritol, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin, processed_5_erythritol, processed_15_erythritol,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Apply PCA for dimensionality reduction if needed
pca = PCA(n_components=20)  # Adjust number of components
X_pca = pca.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Define SVM model
svm = SVC(kernel="linear")

# Apply RFE
rfe = RFE(estimator=svm, n_features_to_select=10)  # Adjust number of features
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train SVM on selected features
svm.fit(X_train_rfe, y_train)

# Predict on test set
y_pred = svm.predict(X_test_rfe)

# Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot heatmap of confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM with RFE and Augmentation")
plt.show()

#relif algorithm + svm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from skrebate import ReliefF  # Make sure to install skrebate with pip install skrebate
from sklearn.svm import SVC

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for SVM processing
X = []
y = []

# Assuming these are your pre-processed images; replace with your actual variable names
for label, img in enumerate([processed_stevia, processed_erythritol, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin, processed_5_erythritol, processed_15_erythritol,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define SVM model
svm = SVC(kernel="linear")

# Apply ReliefF for feature selection
relief = ReliefF(n_features_to_select=10)  # Adjust number of features
X_train_reduced = relief.fit_transform(X_train, y_train)
X_test_reduced = relief.transform(X_test)

# Train SVM on selected features
svm.fit(X_train_reduced, y_train)

# Predict on test set
y_pred = svm.predict(X_test_reduced)

# Evaluate model
print("Confusion Matrix: Relief Algrithm and SVM")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot heatmap of confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM with ReliefF and Augmentation")
plt.show()

#information gain + svm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.svm import SVC

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for SVM processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_erythritol, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin, processed_5_erythritol, processed_15_erythritol,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define SVM model
svm = SVC(kernel="linear")

# Apply Information Gain for feature selection
info_gain = SelectKBest(score_func=mutual_info_classif, k=10)  # Adjust the number of features to select
X_train_reduced = info_gain.fit_transform(X_train, y_train)
X_test_reduced = info_gain.transform(X_test)

# Train SVM on selected features
svm.fit(X_train_reduced, y_train)

# Predict on test set
y_pred = svm.predict(X_test_reduced)

# Evaluate model
print("Confusion Matrix: Information Gain and SVM")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot heatmap of confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM with Information Gain and Augmentation")
plt.show()

#info gain 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Information Gain for feature selection
info_gain = SelectKBest(score_func=mutual_info_classif, k=10)  # Adjust the number of features to select
X_train_reduced = info_gain.fit_transform(X_train, y_train)
X_test_reduced = info_gain.transform(X_test)

# Define and train models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

for model_name, model in models.items():
    # Train and predict
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Information Gain and Augmentation")
    plt.show()


#fisher score = linear, rbf, rf, multi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to calculate Fisher Score manually
def fisher_score(X, y):
    scores = []
    unique_classes = np.unique(y)
    
    for feature in range(X.shape[1]):
        numerator = 0
        denominator = 0
        for c in unique_classes:
            X_c = X[y == c, feature]
            mean_c = np.mean(X_c)
            variance_c = np.var(X_c)
            n_c = len(X_c)
            mean_feature = np.mean(X[:, feature])
            numerator += n_c * (mean_c - mean_feature) ** 2
            denominator += n_c * variance_c
        scores.append(numerator / denominator if denominator != 0 else 0)
    
    return np.array(scores)

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    return [rotate(image, angle, mode='reflect', preserve_range=True, resize=True) for angle in angles]

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        X.append(aug_img.reshape(-1, aug_img.shape[2]))
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Fisher Score for feature selection
scores = fisher_score(X_train, y_train)
k = 10  # Select the top 10 features
selected_features = np.argsort(scores)[-k:]
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Define and train RBF SVM model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train_reduced, y_train)
y_pred = model.predict(X_test_reduced)

# Display classification report
print("RBF SVM - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=sample_names))

# Plot heatmap of confusion matrix with sample names
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=sample_names, yticklabels=sample_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - RBF SVM with Fisher Score")
plt.show()


#dct
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate, resize
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample labels for display in confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", 
                "5% Maltodextrin", "15% Maltodextrin"]

# Set a consistent image size for all samples
target_size = (64, 64)  # Define a suitable size for resizing

# Function to apply DCT and flatten the coefficients
def extract_dct_features(image, num_coefficients=100):
    # Apply DCT on each channel separately, then flatten and limit to num_coefficients
    dct_features = []
    for i in range(image.shape[2]):  # Loop over spectral channels if image has depth
        dct_channel = dct(dct(dct(image[:, :, i], axis=0, norm='ortho'), axis=1, norm='ortho'), norm='ortho')
        # Flatten and select only the first num_coefficients DCT coefficients
        dct_features.extend(dct_channel.flatten()[:num_coefficients])
    return np.array(dct_features)

# Function to rotate and resize images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True)
        resized_image = resize(rotated_image, target_size, mode='reflect', anti_aliasing=True)
        augmented_images.append(resized_image)
    return augmented_images

# Augment each processed image, extract DCT features, and prepare for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        features = extract_dct_features(aug_img)  # Extract DCT features
        X.append(features)
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel="linear", random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix with sample names
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with DCT and Augmentation")
    plt.show()

#fsmrmr + multi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply FSMRMR for feature selection
fsmrmr_selector = SelectKBest(score_func=mutual_info_classif, k=10)  # Adjust number of features to select
X_train_reduced = fsmrmr_selector.fit_transform(X_train, y_train)
X_test_reduced = fsmrmr_selector.transform(X_test)

# Define and train the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
model.fit(X_train_reduced, y_train)
y_pred = model.predict(X_test_reduced)

# Print confusion matrix and classification report
print("Multinomial Logistic Regression - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=sample_names))

# Plot heatmap of confusion matrix with sample names on x and y axes
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=sample_names, yticklabels=sample_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Multinomial Logistic Regression with FSMRMR and Augmentation")
plt.show()

#variance thresholding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables, excluding erythritol and its mixtures
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Variance Threshold for feature selection
var_thresh = VarianceThreshold(threshold=0.01)  # Set threshold based on your data
X_train_reduced = var_thresh.fit_transform(X_train)
X_test_reduced = var_thresh.transform(X_test)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Variance Thresholding and Augmentation")
    plt.show()
    
#chi square
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables, excluding erythritol and its mixtures
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Chi-Square test for feature selection
chi2_selector = SelectKBest(chi2, k=10)  # Adjust `k` to the number of best features you want
X_train_reduced = chi2_selector.fit_transform(X_train, y_train)
X_test_reduced = chi2_selector.transform(X_test)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train_reduced, y_train) 
    y_pred = model.predict(X_test_reduced)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Chi-Square and Augmentation")
    plt.show()


#lbp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to compute LBP and flatten for ML processing
def extract_lbp_features(image, P=8, R=1):
    # Convert to LBP and flatten
    lbp = local_binary_pattern(image, P, R, method="uniform")
    return lbp.flatten()

# Function to rotate images at specified angles, resize, and apply LBP
def augment_and_extract_features(image, angles=[0, 90, 180], target_size=(64, 64), P=8, R=1):
    features = []
    for angle in angles:
        # Rotate the image without resizing
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True)
        # Resize to the target size
        resized_image = resize(rotated_image, target_size, anti_aliasing=True, preserve_range=True)
        # Extract LBP features
        lbp_features = extract_lbp_features(resized_image[..., 0], P, R)  # Apply LBP to a single channel
        features.append(lbp_features)
    return features

# Prepare data arrays
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_features = augment_and_extract_features(img)
    for feature in augmented_features:
        X.append(feature)  # Append LBP feature vector
        y.append(label)  # Append corresponding label

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with LBP and Augmentation")
    plt.show()
    
    
#fractal analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function for fractal dimension feature extraction using box-counting method
def box_counting(image, threshold=0.9):
    # Average across all channels to get a single 2D image
    if image.ndim == 3 and image.shape[2] > 3:
        image = np.mean(image, axis=2)  # Average over channels
    
    # Apply binary thresholding
    image = (image > threshold * np.max(image)).astype(np.uint8)
    sizes = np.arange(1, min(image.shape) // 2, 2)
    counts = []

    # Count the number of boxes for each box size
    for size in sizes:
        count = (image[:image.shape[0] // size * size, :image.shape[1] // size * size]
                 .reshape(image.shape[0] // size, size, -1, size)
                 .any(axis=(1, 3))
                 .sum())
        counts.append(count)

    # Calculate the fractal dimension (slope of log-log plot)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]

# Function to rotate images at specified angles, resize, and apply fractal feature extraction
def augment_and_extract_fractal_features(image, angles=[0, 90, 180], target_size=(64, 64)):
    features = []
    for angle in angles:
        # Rotate the image without resizing
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True)
        # Resize to the target size
        resized_image = resize(rotated_image, target_size, anti_aliasing=True, preserve_range=True)
        # Extract fractal feature
        fractal_feature = box_counting(resized_image)
        features.append([fractal_feature])
    return features

# Prepare data arrays
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_features = augment_and_extract_fractal_features(img)
    for feature in augmented_features:
        X.append(feature)  # Append fractal feature vector
        y.append(label)    # Append corresponding label

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "RBF SVM": SVC(kernel='rbf', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Train and evaluate each model within a pipeline that handles NaNs
for model_name, model in models.items():
    # Define a pipeline with imputation and model training
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', model)
    ])
    
    # Train and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Fractal Feature and Augmentation")
    plt.show()
    
#lda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True, resize=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Flatten each augmented image and add to X
        X.append(aug_img.reshape(-1, aug_img.shape[2]))  # Flatten spatial and spectral dimensions
        # Add corresponding labels for each pixel in the flattened image
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

# Convert lists to numpy arrays
X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=len(np.unique(y)) - 1)
X_train_reduced = lda.fit_transform(X_train, y_train)
X_test_reduced = lda.transform(X_test)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with LDA and Augmentation")
    plt.show()

#morpho
import numpy as np
from skimage.morphology import opening, closing, disk
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary without erythritol or its mixtures
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Morphological Profile function for spatial feature extraction
def morphological_profile(image, selem=disk(1)):
    profile_features = []
    h, w, bands = image.shape
    
    for band in range(bands):
        band_image = image[:, :, band]
        
        # Apply opening and closing
        opened = opening(band_image, selem)
        closed = closing(band_image, selem)
        
        # Append the results to profile features
        profile_features.append(opened)
        profile_features.append(closed)
    
    return np.stack(profile_features, axis=2)

# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, resize=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        mp_image = morphological_profile(aug_image)  # Extract spatial features
        h, w, features = mp_image.shape
        X.append(mp_image.reshape(h * w, features))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                  # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spatial_features)
y = np.array(y)      # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=list(data.keys())))
    
    # Plot heatmap of confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
                xticklabels=list(data.keys()), yticklabels=list(data.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name} with Morphological Profiles')
    plt.show()
    
#spatial wavelet
import numpy as np
import pywt  # For wavelet transformation
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Original data dictionary without erythritol or its mixtures
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Wavelet analysis function for spatial feature extraction
def wavelet_analysis(image, wavelet='haar', level=2):
    spatial_features = []
    h, w, bands = image.shape
    
    for band in range(bands):
        band_image = image[:, :, band]
        
        # Perform wavelet decomposition for each 2D spatial slice
        coeffs = pywt.wavedec2(band_image, wavelet, level=level)
        
        # Extract approximation coefficients as the main spatial feature
        # Optionally, other coefficients (detail) can be added for more spatial texture information
        approx_coeffs = coeffs[0]  # approximation coefficients
        
        # Reshape approximation coefficients to maintain spatial information
        spatial_features.append(approx_coeffs)
    
    # Stack features along the third axis to retain (height, width, bands) structure
    return np.stack(spatial_features, axis=2)


# Data augmentation function to rotate images at 0°, 90°, and 180°
def augment_image(image):
    rotations = [0, 90, 180]
    augmented_images = [rotate(image, angle, resize=False) for angle in rotations]
    return augmented_images

# Flatten each augmented image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_image(image)  # Apply data augmentation
    for aug_image in augmented_images:
        wa_image = wavelet_analysis(aug_image)  # Extract spatial features with wavelet analysis
        h, w, features = wa_image.shape
        X.append(wa_image.reshape(h * w, features))  # Flatten spatial dimensions
        y.extend([label] * (h * w))                 # Assign label for each pixel

# Convert lists to numpy arrays
X = np.vstack(X)     # Shape: (total_samples, spatial_features)
y = np.array(y)      # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train and predict
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=list(data.keys())))
    
    # Plot heatmap of confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(data.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
                xticklabels=list(data.keys()), yticklabels=list(data.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name} with Wavelet Analysis')
    plt.show()
    
#spectral wavelet
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pywt  # PyWavelets for wavelet transforms

# Original data dictionary (removed erythritol and its mixtures)
data = {
    'pure stevia': processed_stevia,
    'pure maltodextrin': processed_maltodextrin,
    'pure saccharin': processed_saccharin,
    '5% saccharin': processed_5_saccharin,
    '5% maltodextrin': processed_5_maltodextrin,
    '15% saccharin': processed_15_saccharin,
    '15% maltodextrin': processed_15_maltodextrin
}

# Function to augment data by rotating images
def augment_data(image):
    augmented_images = [image]  # Start with the original image
    for angle in [90, 180]:  # Use fewer rotations to save memory
        rotated_image = rotate(image, angle, reshape=False)  # Rotate without reshaping
        augmented_images.append(rotated_image)
    return augmented_images

# Function to apply wavelet transform with downsampling
def apply_wavelet_transform(image, wavelet='db1', target_length=None):
    h, w, bands = image.shape
    wavelet_transformed = []
    
    for b in range(bands):
        # Apply wavelet transform on each spectral band
        coeffs = pywt.dwt2(image[:, :, b], wavelet)
        cA, (cH, cV, cD) = coeffs  # Only keep the approximation coefficients
        downsampled_cA = cA.flatten()[::4]  # Downsample: keep every 4th coefficient
        wavelet_transformed.append(downsampled_cA)
    
    full_vector = np.concatenate(wavelet_transformed)
    
    # If target length is specified, pad or trim to match the length
    if target_length and len(full_vector) < target_length:
        full_vector = np.pad(full_vector, (0, target_length - len(full_vector)), 'constant')
    
    return full_vector

# Determine maximum length of transformed feature vector to ensure uniformity
max_length = 0
for label, image in data.items():
    for aug_image in augment_data(image):
        transformed_image = apply_wavelet_transform(aug_image)
        max_length = max(max_length, len(transformed_image))

# Flatten each image and create labels
X = []
y = []
for label, image in data.items():
    augmented_images = augment_data(image)
    for aug_image in augmented_images:
        transformed_image = apply_wavelet_transform(aug_image, target_length=max_length)
        X.append(transformed_image)  # Add transformed image to features list
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X, dtype=np.float32)  # Shape: (total_samples, transformed_features)
y = np.array(y)  # Shape: (total_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RBF SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train Multinomial Regression
multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Function to plot confusion matrix with sample names
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(data.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=list(data.keys()), yticklabels=list(data.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Plot confusion matrix for RBF SVM
plot_confusion_matrix(y_test, y_pred_svm, model_name="RBF SVM")

# Plot confusion matrix for Multinomial Regression
plot_confusion_matrix(y_test, y_pred_multi, model_name="Multinomial Regression")

# Plot confusion matrix for Random Forest
plot_confusion_matrix(y_test, y_pred_rf, model_name="Random Forest")

# Print classification reports
print("Classification Report for RBF SVM:")
print(classification_report(y_test, y_pred_svm))

print("\nClassification Report for Multinomial Regression:")
print(classification_report(y_test, y_pred_multi))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

#fisher plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", "15% Saccharin", "5% Saccharin", "5% Maltodextrin", "15% Maltodextrin"]

# Function to calculate Fisher Score manually
def fisher_score(X, y):
    scores = []
    unique_classes = np.unique(y)
    
    for feature in range(X.shape[1]):
        numerator = 0
        denominator = 0
        for c in unique_classes:
            X_c = X[y == c, feature]
            mean_c = np.mean(X_c)
            variance_c = np.var(X_c)
            n_c = len(X_c)
            mean_feature = np.mean(X[:, feature])
            numerator += n_c * (mean_c - mean_feature) ** 2
            denominator += n_c * variance_c
        scores.append(numerator / denominator if denominator != 0 else 0)
    
    return np.array(scores)

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    return [rotate(image, angle, mode='reflect', preserve_range=True, resize=True) for angle in angles]

# Augment each processed image and flatten for ML processing
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        X.append(aug_img.reshape(-1, aug_img.shape[2]))
        y.extend([label] * aug_img.shape[0] * aug_img.shape[1])

X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Fisher Score for feature selection
scores = fisher_score(X_train, y_train)
k = 10  # Select the top 10 features
selected_features = np.argsort(scores)[-k:]
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Plot Fisher Scores for all features
plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores, color="skyblue")
plt.xlabel("Feature Index")
plt.ylabel("Fisher Score")
plt.title("Fisher Scores for Each Feature")

# Highlight the top k features
for idx in selected_features:
    plt.bar(idx, scores[idx], color="orange")

plt.show()

# Define and train RBF SVM model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train_reduced, y_train)
y_pred = model.predict(X_test_reduced)

# Display classification report
print("RBF SVM - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=sample_names))

# Plot heatmap of confusion matrix with sample names
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=sample_names, yticklabels=sample_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - RBF SVM with Fisher Score")
plt.show()

#spatial fisher
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", 
                "15% Saccharin", "5% Saccharin", 
                "5% Maltodextrin", "15% Maltodextrin"]

# Function to calculate Fisher Score manually
def fisher_score(X, y):
    scores = []
    unique_classes = np.unique(y)
    for feature in range(X.shape[1]):
        numerator = 0
        denominator = 0
        for c in unique_classes:
            X_c = X[y == c, feature]
            mean_c = np.mean(X_c)
            variance_c = np.var(X_c)
            n_c = len(X_c)
            mean_feature = np.mean(X[:, feature])
            numerator += n_c * (mean_c - mean_feature) ** 2
            denominator += n_c * variance_c
        scores.append(numerator / denominator if denominator != 0 else 0)
    return np.array(scores)

# Function to extract patches from the image
def extract_patches(image, patch_size=5):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch.flatten())  # Flatten patch to a vector
    return np.array(patches)

# Augment each processed image and extract patches
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    patches = extract_patches(img, patch_size=5)  # Extract patches
    X.append(patches)
    y.extend([label] * len(patches))

X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Fisher Score for spatial feature selection
scores = fisher_score(X_train, y_train)
k = 50  # Select the top 50 spatial features (from patches)
selected_features = np.argsort(scores)[-k:]
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Plot Fisher Scores for all features
plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores, color="skyblue")
plt.xlabel("Feature Index")
plt.ylabel("Fisher Score")
plt.title("Fisher Scores for Spatial Features")

# Highlight the top k features
for idx in selected_features:
    plt.bar(idx, scores[idx], color="orange")

plt.show()

# Define and train RBF SVM model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train_reduced, y_train)
y_pred = model.predict(X_test_reduced)

# Display classification report
print("RBF SVM - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=sample_names))

# Plot heatmap of confusion matrix with sample names
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=sample_names, yticklabels=sample_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - RBF SVM with Spatial Feature Selection")
plt.show()

#spatial fiaher all
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", 
                "15% Saccharin", "5% Saccharin", 
                "5% Maltodextrin", "15% Maltodextrin"]

# Function to calculate Fisher Score manually
def fisher_score(X, y):
    scores = []
    unique_classes = np.unique(y)
    for feature in range(X.shape[1]):
        numerator = 0
        denominator = 0
        for c in unique_classes:
            X_c = X[y == c, feature]
            mean_c = np.mean(X_c)
            variance_c = np.var(X_c)
            n_c = len(X_c)
            mean_feature = np.mean(X[:, feature])
            numerator += n_c * (mean_c - mean_feature) ** 2
            denominator += n_c * variance_c
        scores.append(numerator / denominator if denominator != 0 else 0)
    return np.array(scores)

# Function to extract patches from the image
def extract_patches(image, patch_size=5):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch.flatten())  # Flatten patch to a vector
    return np.array(patches)

# Function to augment data with rotations at 0°, 90°, and 180°
def augment_data(image):
    augmented_images = [rotate(image, angle, resize=False) for angle in [0, 90, 180]]
    return augmented_images

# Augment each processed image and extract patches
X = []
y = []

# Replace these with your actual preprocessed image variables
processed_images = [processed_stevia, processed_maltodextrin, processed_saccharin,
                    processed_15_saccharin, processed_5_saccharin,
                    processed_5_maltodextrin, processed_15_maltodextrin]

for label, img in enumerate(processed_images):
    augmented_imgs = augment_data(img)  # Generate augmented images
    for augmented_img in augmented_imgs:
        patches = extract_patches(augmented_img, patch_size=5)  # Extract patches
        X.append(patches)
        y.extend([label] * len(patches))

X = np.vstack(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Fisher Score for spatial feature selection
scores = fisher_score(X_train, y_train)
k = 50  # Select the top 50 spatial features (from patches)
selected_features = np.argsort(scores)[-k:]
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Plot Fisher Scores for all features
plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores, color="skyblue")
plt.xlabel("Feature Index")
plt.ylabel("Fisher Score")
plt.title("Fisher Scores for Spatial Features")

# Highlight the top k features
for idx in selected_features:
    plt.bar(idx, scores[idx], color="orange")

plt.show()

# Define models
models = {
    "Multinomial Logistic Regression": LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42, max_iter=500),
    "RBF SVM": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining and Evaluating {model_name}...")
    
    # Train the model
    model.fit(X_train_reduced, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_reduced)
    
    # Display classification report
    print(f"{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix with sample names
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Fisher Score (Spatial Selection)")
    plt.show()
    
    
#IG spatial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sample names for the confusion matrix
sample_names = ["Stevia", "Maltodextrin", "Saccharin", 
                "15% Saccharin", "5% Saccharin", 
                "5% Maltodextrin", "15% Maltodextrin"]

# Function to rotate images at specified angles
def augment_image(image, angles=[0, 90, 180]):
    augmented_images = []
    for angle in angles:
        rotated_image = rotate(image, angle, mode='reflect', preserve_range=True)
        augmented_images.append(rotated_image)
    return augmented_images

# Define a fixed size for resizing
fixed_height, fixed_width = 100, 100

# Prepare dataset with fixed spatial size
X = []
y = []

# Replace these with your actual preprocessed image variables
for label, img in enumerate([processed_stevia, processed_maltodextrin, processed_saccharin,
                             processed_15_saccharin, processed_5_saccharin,
                             processed_5_maltodextrin, processed_15_maltodextrin]):
    # Augment images at 0, 90, and 180 degrees
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        # Resize image to fixed dimensions
        resized_img = resize(aug_img, (fixed_height, fixed_width), mode='reflect', preserve_range=True)
        
        # Flatten spatial dimensions and use only the first spectral band
        spatial_data = resized_img[:, :, 0].flatten()  # Using the first spectral band
        
        # Append to dataset
        X.append(spatial_data)
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Display dataset shapes
print(f"Shape of X: {X.shape}")  # Should be (number_of_samples, fixed_height * fixed_width)
print(f"Shape of y: {y.shape}")  # Should be (number_of_samples,)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Information Gain for feature selection
num_features_to_select = 100  # Adjust the number of features to select based on your data
info_gain_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
X_train_reduced = info_gain_selector.fit_transform(X_train, y_train)
X_test_reduced = info_gain_selector.transform(X_test)

# Define models for classification
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_reduced, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_reduced)
    
    # Print confusion matrix and classification report
    print(f"\n{model_name} - Confusion Matrix and Classification Report:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=sample_names))
    
    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=sample_names, yticklabels=sample_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Information Gain (Spatial Selection)")
    plt.show()

#spatial chi

