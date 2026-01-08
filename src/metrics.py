import os
import random
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def normalize_features(gallery, query, train_mean, norm_type):
    if norm_type == "CL2N":
        gallery = gallery - train_mean
        query = query - train_mean
        gallery /= LA.norm(gallery, 2, 1, keepdims=True)
        query /= LA.norm(query, 2, 1, keepdims=True)
    elif norm_type in ("L2N", "COS"):
        gallery /= LA.norm(gallery, 2, 1, keepdims=True)
        query /= LA.norm(query, 2, 1, keepdims=True)
    elif norm_type == "CCOS":
        train_mean /= LA.norm(train_mean, 2, 0)
        gallery -= train_mean
        query -= train_mean
        gallery /= LA.norm(gallery, 2, 1, keepdims=True)
        query /= LA.norm(query, 2, 1, keepdims=True)
    elif norm_type == "CCOS1":
        train_mean /= LA.norm(train_mean, 1, 0)
        gallery /= LA.norm(gallery, 1, 1, keepdims=True)
        query /= LA.norm(query, 1, 1, keepdims=True)
        gallery -= train_mean
        query -= train_mean
        gallery /= LA.norm(gallery, 1, 1, keepdims=True)
        query /= LA.norm(query, 1, 1, keepdims=True)
    return gallery, query


def fast_mode(a):
    a = a.astype(int)
    return np.argmax(np.bincount(a))


def metric_class_type(
    gallery,
    query,
    train_label,
    test_label,
    shot,
    args,
    train_mean=None,
    norm_type="CL2N",
):
    gallery, query = normalize_features(gallery, query, train_mean, norm_type)

    if args.classifier == "nc":
        if args.median_prototype:
            gallery = np.median(
                gallery.reshape(args.test_way, shot, gallery.shape[-1]), axis=1
            )
        else:
            gallery = gallery.reshape(args.test_way, shot, gallery.shape[-1]).mean(1)

        if norm_type == "COS":
            gallery /= LA.norm(gallery, 2, 1, keepdims=True)

        train_label = train_label[::shot]
        distances = cdist(query, gallery, metric="euclidean")
        predictions = train_label[np.argmin(distances, axis=1)]
        return (predictions == test_label).mean()

    elif args.classifier in ("sa", "gsa"):
        distances_sq = cdist(gallery, query, metric="sqeuclidean")
        distance = np.exp(-distances_sq)
        norm_distance = distance / distance.sum(axis=0, keepdims=True)
        norm_distance = norm_distance.reshape(args.test_way, shot, -1)

        score_per_class = (
            norm_distance.sum(axis=1)
            if args.classifier == "sa"
            else np.prod(norm_distance, axis=1)
        )
        train_label = train_label[::shot]
        predictions = train_label[np.argmax(score_per_class, axis=0)]
        return (predictions == test_label).mean()

    return 0.0


def meta_evaluate(data, train_mean, shot, num_iter, args):
    cl2n_list = []
    for _ in range(num_iter):
        train_data, test_data, train_label, test_label = sample_case(data, shot, args)
        acc = metric_class_type(
            train_data,
            test_data,
            train_label,
            test_label,
            shot,
            args,
            train_mean=train_mean,
            norm_type=args.eval_norm_type,
        )
        cl2n_list.append(acc)

    if args.output_dir:
        np.save(
            os.path.join(args.output_dir, f"meta_evaluate_shot{shot}.npy"),
            np.array(cl2n_list),
        )
    return compute_confidence_interval(cl2n_list)


def sample_case(ld_dict, shot, args):
    sample_class = random.sample(list(ld_dict.keys()), args.test_way)
    first_feat = ld_dict[sample_class[0]][0]
    feat_dim = (
        len(first_feat)
        if isinstance(first_feat, (list, np.ndarray))
        else first_feat.shape[0]
    )

    train_input = np.empty((args.test_way * shot, feat_dim), dtype=np.float32)
    test_input = np.empty((args.test_way * args.test_query, feat_dim), dtype=np.float32)
    train_labels, test_labels = [], []
    train_ptr, test_ptr = 0, 0

    for each_class in sample_class:
        samples = np.array(
            random.sample(ld_dict[each_class], shot + args.test_query), dtype=np.float32
        )
        train_input[train_ptr : train_ptr + shot] = samples[:shot]
        test_input[test_ptr : test_ptr + args.test_query] = samples[shot:]
        train_ptr += shot
        test_ptr += args.test_query
        train_labels.extend([each_class] * shot)
        test_labels.extend([each_class] * args.test_query)

    return train_input, test_input, np.array(train_labels), np.array(test_labels)
