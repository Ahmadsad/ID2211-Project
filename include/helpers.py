import numpy as np
import scipy.ndimage
import scipy.sparse.csgraph
import SimpleITK as sitk
import sklearn.cluster
import sklearn.preprocessing
from skimage.measure import block_reduce
from skimage.filters import sobel, gaussian
import sklearn.metrics as metrics
from matplotlib.cm import ScalarMappable
import networkx as nx
import matplotlib.pyplot as plt


<<<<<<< HEAD
def get_image(filename: str, normalize:bool = False, apply_threshold: bool = True, scaling_factor: int = 12, amplify_edges: bool = False):
    image_dcm = sitk.ReadImage(filename)
=======
def get_image(filename: str, normalize: bool = False, apply_threshold: bool = True, scaling_factor: int = 12, amplify_edges: bool = False):
    image_dcm = sitk.ReadImage('include/data/' + filename)
>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411
    image_array_view = sitk.GetArrayViewFromImage(image_dcm)
    image = image_array_view.squeeze()
    image = np.array(image)

    if apply_threshold:
        # clip values to range -160 <-> 240, which is the intensity values of blood-filled organs.
        image[image < -160] = -1024
        image[image > 240] = -1024

    if normalize:
        # add an offset to 'remove' negative values
        image = np.sqrt((image+np.abs(np.min(image)))**2)
        # normalize
        image = image/np.max(image)

    if amplify_edges:
        noiseless_image = gaussian(image, sigma=1)
        sobel_image = sobel(image)
        sobel_image = (sobel_image > 0.02)

        # replace ones with median value of noiseless image
        # TODO: clarify why -->
        # Not sure but we do want avoiding adding to much new information to the data,
        # maybe the mean value could be good here too. adding ones to a normalized image
        # would change the distribution of the data...
        tmp = np.zeros((sobel_image.shape))
        tmp[sobel_image == 1] = np.median(noiseless_image)
        sobel_image = tmp

        image = sobel_image + noiseless_image

    # scale down image
    image = block_reduce(image, block_size=(
        scaling_factor, scaling_factor), func=np.mean)

    return image


def get_ground_truth_image(filename: str, scaling_factor: int = 12):
    image_dcm = sitk.ReadImage(filename)
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


def get_cluster_image(labels, shape: np.array, cluster: int = None):
    cluster_image = np.copy(labels)

    cluster_image = np.reshape(cluster_image, shape)

    # make sure that cluster 0 does not get confused with default value
    cluster_image += 1

    if cluster is not None:
        cluster_image[cluster_image != cluster + 1] = 0

    return cluster_image


def get_evaluation_scores(cluster_image, ground_truth_image):
    cluster_image = np.copy(cluster_image)
    ground_truth_image = np.copy(ground_truth_image)
    cluster_image[cluster_image != 0] = 1
    ground_truth_image[ground_truth_image != 0] = 1
    ground_truth_image = ground_truth_image.astype(int)

    jaccard = metrics.jaccard_score(
        ground_truth_image.flatten(), cluster_image.flatten())

    precision, sensitivity, fscore, _ = metrics.precision_recall_fscore_support(
        ground_truth_image.flatten(), cluster_image.flatten())
    return jaccard, fscore, precision, sensitivity

<<<<<<< HEAD
def get_dice_coeff(ground_truth, mask):
    '''
    Takes a ground truth mask and a mask, return the Dice coefficient of these masks
    '''
    # convert to simple itk image object
    ground_truth = sitk.GetImageFromArray(ground_truth)
    mask = sitk.GetImageFromArray(mask)
    # after using getImage the images are assigned values 0-255, fix this below
    label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask = mask == np.max(mask)
    ground_truth = sitk.Cast(ground_truth, sitk.sitkUInt8)
    ground_truth = ground_truth == np.max(ground_truth)

    label_overlap_measures_filter.Execute(ground_truth, mask)
    return label_overlap_measures_filter.GetDiceCoefficient()
=======
>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411

def get_hausdorff_dist(ground_truth, mask):
    '''
    Takes a ground truth mask and a mask, return the hausdorff distance of these masks
    '''
    ground_truth = sitk.GetImageFromArray(ground_truth)
    mask = sitk.GetImageFromArray(mask)
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
<<<<<<< HEAD
    # after using getImage the images are assigned values 0-255, fix this below
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask = mask == np.max(mask)
    ground_truth = sitk.Cast(ground_truth, sitk.sitkUInt8)
    ground_truth = ground_truth == np.max(ground_truth)
    
=======

>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411
    hausdorff_distance_image_filter.Execute(ground_truth, mask)

    return hausdorff_distance_image_filter.GetHausdorffDistance()


def plt_img_fignHist(image):
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1, 2, 2)
    uniqe = np.unique(image, return_counts=True)
    plt.plot(uniqe[0], uniqe[1])
    plt.xlabel('values')
    plt.ylabel('counts')
    plt.show()


def print_img_info(image):
<<<<<<< HEAD
    unique = np.unique(image);
    print('The first {} values in the image: \n'.format(len(unique[0:5], unique[0:5])));
    print('The maximum value is: {}, and minimum is: {}'.format(np.max(image), np.min(image)));
    print('The mean value is: {}, and the median is: {}'.format(np.mean(image), np.median(image)));
    print('Total Number of values in the image: ',len(unique));
    print('The image size: {}, dimensions: {} and the shape: {}'.format(image.size, image.ndim, image.shape));    
    
=======
    unique = np.unique(image)
    print('The first 5 values in the image: \n', unique[0:5])
    print('The maximum value is: {}, and minimum is: {}'.format(
        np.max(image), np.min(image)))
    print('The mean value is: {}, and the median is: {}'.format(
        np.mean(image), np.median(image)))
    print('Total Number of values in the image: ', len(unique))
    print('The image size: {}, dimensions: {} and the shape: {}'.format(
        image.size, image.ndim, image.shape))


>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411
def build_graph_of_simiMatrix(simi_matrix):
    return nx.convert_matrix.from_numpy_matrix(simi_matrix)


def get_avg_cluster_coef(graph):
    return nx.average_clustering(graph)


def get_normalizedCut_value(graph, subGraph):
    return nx.normalized_cut_size(graph, subGraph)

<<<<<<< HEAD
def get_general_graph_info(graph):
    inf = nx.info(graph)
    return inf + ' \nDensity of graph: {}'.format(nx.density(graph))
=======

def get_general_graph_info(graph):
    inf = nx.info(gwm_img)
    info['Density og graph'] = nx.density(gwm_img)
    return inf
>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411


def plt_graph_hist(graph):
    hist = nx.degree_histogram(graph)
    plt.plot(range(0, len(hist), 1), hist)
    plt.xlabel('degree values, index in the list')
    plt.ylabel('values of hist, frequencies of degrees')
    plt.show()
    return hist


def plot_cluster_distribuition(graph):
    # https://stackoverflow.com/questions/64485434/how-to-plot-the-distribution-of-a-graphs-clustering-coefficient
    g_connected = graph.subgraph(max(nx.connected_components(graph)))
    list_cluster_coef = nx.clustering(g_connected)

    cmap = plt.get_cmap('autumn')
    norm = plt.Normalize(0, max(list_cluster_coef.values()))
    node_colors = [cmap(norm(list_cluster_coef[node]))
                   for node in g_connected.nodes]
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))
    nx.draw_spring(g_connected, node_color=node_colors,
                   with_labels=False, ax=ax1)
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm),
                 label='Clustering', shrink=0.95, ax=ax1)

    ax2.hist(list_cluster_coef.values(), bins=10)
    ax2.set_xlabel('Clustering')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
<<<<<<< HEAD

def get_subgrapg(graph, clustering_label):
  '''Takes a graph and an array with the indices of the nodes to be separated in a new subgrapg
      return a subgraph with the specified nodes of the given clustering labels'''
  return graph.subgraph(list((clustering_label)[0]))
    
   
    
def plot_multiple_masks(masks, clusters_label, img):
    row=1
    # Figure: Subplot
    fig, axs = plt.subplots(row, len(masks),figsize=(20,20))

    # Plot Data
    for col, mask in enumerate(masks):
        try:
            axs[col].imshow(img, cmap='Reds')
            axs[col].imshow(mask==clusters_label[col], cmap='Blues', alpha=0.6)
        except:
            pass
        axs[col].set_title("Mask {} merged on image".format(col))
    fig.show()
=======
>>>>>>> 9f1f79468d106bf50c564e889f5c7294fe16c411
