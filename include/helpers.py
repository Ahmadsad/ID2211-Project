import numpy as np
import matplotlib.pyplot as plt
from pandas.core.accessor import register_index_accessor
# to save and write as pngs
import scipy.ndimage
import scipy.sparse.csgraph
import SimpleITK as sitk
import sklearn.cluster
import sklearn.preprocessing
from skimage.measure import block_reduce
from skimage.filters import sobel, gaussian


def get_image(filename: str, apply_threshold: bool = True, scaling_factor: int = 12, amplify_edges: bool = False):
    image_dcm = sitk.ReadImage('include/data/' + filename)
    image_array_view = sitk.GetArrayViewFromImage(image_dcm)
    image = image_array_view.squeeze()
    image = np.array(image)

    if apply_threshold:
        # clip values to range -160 <-> 240, which is the intensity values of blood-filled organs.
        image[image < -160] = -1024
        image[image > 240] = -1024

    # add an offset and 'remove' negative values
    image = np.sqrt((image+np.abs(np.min(image)))**2)

    # normalize
    image = image/np.max(image)

    if amplify_edges:
        noiseless_image = gaussian(image, sigma=1)
        sobel_image = sobel(image)
        sobel_image = (sobel_image > 0.02)

        # replace ones with median value of noiseless image
        # TODO: clarify why
        tmp = np.zeros((sobel_image.shape))
        tmp[sobel_image == 1] = np.median(noiseless_image)
        sobel_image = tmp

        image = sobel_image + noiseless_image

    # scale down image
    image = block_reduce(image, block_size=(
        scaling_factor, scaling_factor), func=np.mean)

    return image


def get_ground_truth_image(filename: str, scaling_factor: int = 12):
    image_dcm = sitk.ReadImage('include/data/' + filename)
    image = sitk.GetArrayFromImage(image_dcm)

    # scale down image
    image = block_reduce(image, block_size=(
        scaling_factor, scaling_factor), func=np.mean)

    return image


def build_similarity_matrix(image: np.array, use_spatial: bool = False, spatial_radius: int = 6):
    # similarity matrix - nodes are connected if they are neighbours and they share the same value
    num_pixels = image.size

    similarity_matrix = np.zeros((num_pixels, num_pixels), dtype=float)

    # used for printing progress
    meaningful_segment = int(0.1 * num_pixels)

    image_width = image.shape[0]

    sigma_distance = int(0.1 * image_width)  # 10-20% of feature distance
    sigma_intensity = 0.1  # 10-20% of feature distance (0 <-> 1)

    for i in range(0, image_width):
        for j in range(0, image_width):
            sm_i = j + i * image_width
            if sm_i % meaningful_segment == 0:
                print("{:.2f}%".format((sm_i/num_pixels)*100))
            for ii in range(0, image_width):
                for jj in range(0, image_width):
                    sm_j = jj + ii * image_width
                    # skip already defined values if possible
                    if similarity_matrix[sm_i, sm_j] != 0:
                        continue

                    distance = 0
                    if use_spatial:
                        distance = np.linalg.norm(
                            np.array([i, j]) - np.array([ii, jj]))
                        if distance > spatial_radius:
                            continue

                    intensity_diff = np.linalg.norm(
                        np.array(image[i, j]) - np.array(image[ii, jj]))
                    weight = np.exp(-np.power(intensity_diff,
                                              2)/sigma_intensity)

                    if use_spatial:
                        weight *= np.exp(-np.power(distance, 2)/sigma_distance)

                    similarity_matrix[sm_i, sm_j] = weight
                    # matrix is symmetric => (sm_i, sm_j) = (sm_j, sm_i)
                    similarity_matrix[sm_j, sm_i] = weight

    return similarity_matrix


def get_spectral_clustering(similarity_matrix: np.array, n_clusters: int = 10):
    # clustering following: https://changyaochen.github.io/spectral-clustering/
    graph_laplacian = scipy.sparse.csgraph.laplacian(
        similarity_matrix, normed=True)

    # get eigenvalues + vectors
    eigenvalues, eigenvectors = np.linalg.eigh(graph_laplacian)

    # sort eigenvalues and eigenvectors (ascending order based on eigenvalue)
    ind = np.argsort(eigenvalues, axis=0)
    eigenvalues = eigenvalues[ind]
    eigenvectors = eigenvectors[:, ind]

    # select eigenvectors for the k smallest eigenvalues
    selected_eigenvectors = eigenvectors[:, :n_clusters]

    # normalize
    selected_eigenvectors = sklearn.preprocessing.normalize(
        selected_eigenvectors, norm='l2', axis=1)

    # k-means to get clusters
    clusters = sklearn.cluster.KMeans(
        n_clusters=n_clusters).fit(selected_eigenvectors)

    return clusters.labels_, eigenvalues[:n_clusters], selected_eigenvectors


def plot_clustering(labels, shape: np.array, cluster: int = None):
    cluster_image = np.zeros(shape)

    for i, l in enumerate(labels):
        x = i % shape[0]
        y = int(i / shape[0])
        if cluster is None or l == cluster:
            # make sure that cluster 0 does not get confused with default value
            cluster_image[y, x] = l + 1

    # plot image
    plt.imshow(cluster_image, cmap="gray")
    plt.show()
