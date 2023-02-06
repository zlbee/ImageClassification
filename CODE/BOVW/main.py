import argparse
import configparser
import json
import pickle

import cv2
import numpy as np

import auxiliary
import feature_extraction
import TFIDF
import classifiers
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import image_representation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default='sift_kmeans_rf.config', help='Configuration file')
    parser.add_argument('--train_mode', type=str, required=True, default='train', help='Train Mode')
    args = parser.parse_args()

    config_path = args.config
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_path)

    # read dataset
    x_train, y_train, x_test, y_test = None, None, None, None
    if config['METHOD']['dataset'] == 'cifar':
        x_train, y_train, x_test, y_test = auxiliary.load_CIFAR10(config['PATH']['CIFAR10_path'])
    elif config['METHOD']['dataset'] == 'stl':
        x_train, y_train, x_test, y_test = auxiliary.read_STL10()


    if args.train_mode == 'train':
        if config['METHOD']['dataset'] == 'cifar':
            # decode images
            x_train = auxiliary.decode_imgs(x_train)
            x_test = auxiliary.decode_imgs(x_test)

        # step 1: feature extraction
        all_descriptors_images_train = None
        all_descriptors_images_test = None
        all_descriptors_test = None
        all_descriptors_train = None

        # # convert to gray
        x_train_gray = np.zeros(x_train.shape[:-1])
        for i in range(x_train.shape[0]):
            x_train_gray[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY)
        # # convert data type
        x_train_gray = x_train_gray.astype(np.uint8)

        # # convert to gray
        x_test_gray = np.zeros(x_test.shape[:-1])
        for i in range(x_test.shape[0]):
            x_test_gray[i] = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2GRAY)
        # # convert data type
        x_test_gray = x_test_gray.astype(np.uint8)

        if config['METHOD']['feature_extraction'] == 'sift':
            # # SIFT
            # # SIFT train
            all_descriptors_train, all_descriptors_images_train = feature_extraction.find_sifts_images(x_train_gray)
            # # SIFT test
            all_descriptors_test, all_descriptors_images_test = feature_extraction.find_sifts_images(x_test_gray)
        elif config['METHOD']['feature_extraction'] == 'surf':
            # # SURF
            # # SURF train
            all_descriptors_train, all_descriptors_images_train = feature_extraction.find_surfs_images(x_train_gray,
                                                                                                       run_mode='train',
                                                                                                       train_mode='train')
            # # SIFT test
            all_descriptors_test, all_descriptors_images_test = feature_extraction.find_surfs_images(x_test_gray,
                                                                                                     run_mode='train',
                                                                                                     train_mode='test')
        elif config['METHOD']['feature_extraction'] == 'orb':
            # # ORB
            # # ORB train
            all_descriptors_train, all_descriptors_images_train = feature_extraction.find_orbs_images(x_train_gray,
                                                                                                       run_mode='train',
                                                                                                       train_mode='train')
            # # ORB test
            all_descriptors_test, all_descriptors_images_test = feature_extraction.find_orbs_images(x_test_gray,
                                                                                                     run_mode='train',
                                                                                                     train_mode='test')

        # PCA
        # all_descriptors_train = StandardScaler().fit_transform(all_descriptors_train)
        # pca = PCA()
        # all_descriptors_train = pca.fit_transform(all_descriptors_train)
        #
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('number of components')
        # plt.ylabel('cumulative explained variance')
        # plt.show()


        # for i, img in enumerate(all_descriptors_images_train):
        #     if img is not None:
        #         all_descriptors_images_train[i] = pca.transform(img)
        # for i, img in enumerate(all_descriptors_images_test):
        #     if img is not None:
        #         all_descriptors_images_test[i] = pca.transform(img)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xvis_data = all_descriptors_train[:, 0]
        # yvis_data = all_descriptors_train[:, 1]
        # zvis_data = all_descriptors_train[:, 2]
        # ax.scatter3D(xvis_data, yvis_data, zvis_data)
        # plt.show()


        # step 2: image representation
        clustering, len_unique_labels = None, None
        if config['METHOD']['image_rep'] == 'kmeans':
            # image_representation.cluster_Kmeans_elbow(all_descriptors_train)
            clustering, len_unique_labels = image_representation.cluster_vws_batchKmeans(
                int(config['HYPER_PARAMETER']['clusters']), all_descriptors_train)
        elif config['METHOD']['image_rep'] == 'dbscan':
            clustering, len_unique_labels = image_representation.cluster_vws_DBSCAN(
                int(config['HYPER_PARAMETER']['clusters']), all_descriptors_train)
        elif config['METHOD']['image_rep'] == 'birch':
            clustering, len_unique_labels = image_representation.cluster_vws_BIRCH(
                int(config['HYPER_PARAMETER']['clusters']), all_descriptors_train)
        elif config['METHOD']['image_rep'] == 'ms':
            clustering, len_unique_labels = image_representation.cluster_vws_MS(
                int(config['HYPER_PARAMETER']['clusters']), all_descriptors_train)
        elif config['METHOD']['image_rep'] == 'gm':
            clustering, len_unique_labels = image_representation.cluster_vws_GM(
                int(config['HYPER_PARAMETER']['clusters']), all_descriptors_train)

        # create histogram for images
        histograms_train = image_representation.create_histograms(all_descriptors_images_train, len_unique_labels, clustering)
        histograms_test = image_representation.create_histograms(all_descriptors_images_test, len_unique_labels, clustering)

        # save histograms
        with open('./intermediate/hist_train_'+config_path.replace('.', '')+'.pickle', 'wb') as fp:
            pickle.dump(histograms_train, fp)
        fp.close()
        with open('./intermediate/hist_test_'+config_path.replace('.', '')+'.pickle', 'wb') as fp:
            pickle.dump(histograms_test, fp)
        fp.close()

    # read histograms
    with open('./intermediate/hist_train_'+config_path.replace('.', '')+'.pickle', 'rb') as fp:
        histograms_train = pickle.load(fp)
    fp.close()
    with open('./intermediate/hist_test_'+config_path.replace('.', '')+'.pickle', 'rb') as fp:
        histograms_test = pickle.load(fp)
    fp.close()

    # apply TF-IDF
    if config['HYPER_PARAMETER']['TFIDF'] == 'True':
        hist_train_matrix = TFIDF.list2csr_matrix(histograms_train)
        hist_test_matrix = TFIDF.list2csr_matrix(histograms_test)
        tfidf_train = TFIDF.calc_TFIDF(hist_train_matrix).toarray()
        tfidf_test = TFIDF.calc_TFIDF(hist_test_matrix).toarray()
        histograms_train = np.multiply(np.asarray(histograms_train), tfidf_train)
        histograms_test = np.multiply(np.asarray(histograms_test), tfidf_test)
        print('TFIDF finished.')
        pass

    # train classifier
    # # choose classifier
    clf = None
    if config['METHOD']['clf'] == 'SVM':
        clf = classifiers.clf1
    elif config['METHOD']['clf'] == 'KNN':
        clf = classifiers.clf2
    elif config['METHOD']['clf'] == 'RF':
        clf = classifiers.clf3
    # elif config['CLASSIFIER']['clf'] == 'GPC':
    #     clf = classifiers.clf4

    # # fit classifier
    clf.fit(histograms_train, y_train)

    # # predict
    y_pred = clf.predict(histograms_test)

    # # evaluate
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    print('Accuracy: ', acc)
    print('F1 Score: ', f1)
    with open('./intermediate/result/score_'+config_path.replace('.', '')+'.json', 'w') as f:
        json.dump([acc, f1], f)

    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)
    plt.savefig('./intermediate/result/cm_'+config_path.replace('.', '')+'.jpg')
    plt.show()
    
    pass
