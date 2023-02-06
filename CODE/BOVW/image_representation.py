import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# step 2: image representation
# # KMeans
def cluster_Kmeans_elbow(descriptors_imgs):
    inertia_list = []
    for num_clusters in range(5, 100, 3):
        kmeans = KMeans(n_clusters=num_clusters, verbose=1, max_iter=300, n_init=1, init='k-means++')
        kmeans.fit(descriptors_imgs)
        inertia_list.append(kmeans.inertia_)

    # plot the inertia curve
    plt.plot(range(5, 100, 3), inertia_list)
    plt.scatter(range(5, 100, 3), inertia_list)
    plt.scatter(3, inertia_list[3], marker="X", s=300, c="r")
    plt.xlabel("Number of Clusters", size=13)
    plt.ylabel("Inertia Value", size=13)
    plt.title("Different Inertia Values for Different Number of Clusters", size=17)
    plt.show()

def cluster_vws_Kmeans(K, descriptors_imgs):
    print('Kmeans clustering started.')
    kmeans = KMeans(n_clusters=K, random_state=1, verbose=1, max_iter=600, n_init=20, init='random', tol=1e-3)
    kmeans.fit(descriptors_imgs)
    # visual_words = kmeans.cluster_centers_
    len_unique_labels = len(np.unique(kmeans.labels_))
    print('Kmeans clustering finished.')

    # plot clusters
    # labels = kmeans.fit_predict(descriptors_imgs)
    # # unique_labels = np.unique(labels)
    # # for i in unique_labels:
    # plt.scatter(descriptors_imgs[:, 0], descriptors_imgs[:, 1])
    # plt.scatter(kmeans.cluster_centers_[:, 0],
    #             kmeans.cluster_centers_[:, 1],
    #             c='red')  # Set centroid color
    # plt.legend()
    # plt.show()
    return kmeans, len_unique_labels


# mini batch KMeans
def cluster_vws_batchKmeans(K, descriptors):
    print('Kmeans clustering started.')

    # fit model
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=1, batch_size=256, verbose=1)
    kmeans.fit(descriptors)
    len_unique_labels = len(np.unique(kmeans.labels_))
    print('Kmeans clustering finished.')
    return kmeans, len_unique_labels


# # DBSCAN
def cluster_vws_DBSCAN(n_clusters, descriptors_img):
    print('DBSCAN started.')
    # DBSCAN
    dbscan = DBSCAN(n_jobs=-1)
    labels = dbscan.fit_predict(descriptors_img)
    len_unique_labels = len(np.unique(labels))
    print('DBSCAN ended.')
    return dbscan, len_unique_labels


# # BIRCH
def cluster_vws_BIRCH(n_clusters, descriptors_img):
    print('BIRCH started.')
    birch = Birch(n_clusters=n_clusters)
    birch.fit(descriptors_img)
    len_unique_labels = len(np.unique(birch.labels_))
    print('BIRCH ended.')
    return birch, len_unique_labels


# # Mean Shift
def cluster_vws_MS(n_clusters, descriptors_img):
    print('MS started.')
    ms = MeanShift(n_jobs=2)
    ms.fit(descriptors_img)
    len_unique_labels = len(np.unique(ms.labels_))
    print(len_unique_labels)
    print('MS ended.')
    return ms, len_unique_labels

# # Gaussian Mixture
def cluster_vws_GM(n_clusters, descriptors_img):
    print('GM started.')
    gm = GaussianMixture(n_components=n_clusters, random_state=1, verbose=1, max_iter=150)
    gm.fit(descriptors_img)
    len_unique_labels = n_clusters
    print('GM ended.')
    return gm, len_unique_labels


# step 3: aggregate histogram from image representations
def create_histograms(descriptors_images, len_centroids, clustering):
    print('Histograms calculation started.')
    histograms = []
    for descriptors_image in descriptors_images:
        histogram = np.zeros(len_centroids)
        if descriptors_image is None:
            histograms.append(histogram)
            continue
        labels_image = clustering.predict(descriptors_image.astype(np.double))
        for label in labels_image:
            histogram[label] += 1
        histograms.append(histogram)
    print('Histograms calculation ended.')
    histograms = np.asarray(histograms)
    return histograms