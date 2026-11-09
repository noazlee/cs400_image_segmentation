import numpy as np

def dice(ground_truth, prediction, num_classes=4, ignore_id=None):
    gt_pixels = np.asarray(ground_truth).flatten()
    pred_pixels = np.asarray(prediction).flatten()

    if len(gt_pixels) != len(pred_pixels):
        raise ValueError("Input images must have the same number of pixels.")

    # AI4Mars uses 255 for unlabeled / NULL pixels.
    valid = (gt_pixels != 255)
    gt_pixels = gt_pixels[valid]
    pred_pixels = pred_pixels[valid]

    dice_scores = []

    # Calculate Dice separately for every semantic class, then mean them.
    for class_id in range(num_classes):
        if ignore_id == class_id:
            continue

        gt_class = (gt_pixels == class_id)
        pred_class = (pred_pixels == class_id)

        intersection = np.sum(gt_class & pred_class)
        gt_size = np.sum(gt_class)
        pred_size = np.sum(pred_class)

        denominator = gt_size + pred_size

        # If the class is absent from both masks, do not include it in the mean.
        if denominator > 0:
            dice_scores.append((2 * intersection) / denominator)

    return float(np.mean(dice_scores)) if dice_scores else np.nan


def iou(ground_truth, prediction, num_classes=4, ignore_id=None):
    gt_pixels = np.asarray(ground_truth).flatten()
    pred_pixels = np.asarray(prediction).flatten()

    if len(gt_pixels) != len(pred_pixels):
        raise ValueError("Input images must have the same number of pixels.")

    valid = (gt_pixels != 255)
    gt_pixels = gt_pixels[valid]
    pred_pixels = pred_pixels[valid]

    ious = []

    # Calculate IoU separately for every semantic class, then mean them.
    for class_id in range(num_classes):
        if ignore_id == class_id:
            continue
        
        gt_class = (gt_pixels == class_id)
        pred_class = (pred_pixels == class_id)

        intersection = np.sum(gt_class & pred_class)
        union = np.sum(gt_class | pred_class)

        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else np.nan

# Pixel proportions of the regular test set (data_preprocessing/proportions.py):
# soil, bedrock, sand, big rock. Unlabelled pixels are left out.
CLASS_PROPORTIONS = np.array([0.20658, 0.27078, 0.06877, 0.00465])

def weighted_iou(ground_truth, prediction, num_classes=4, ignore_id=None):
    gt_pixels = np.asarray(ground_truth).flatten()
    pred_pixels = np.asarray(prediction).flatten()

    if len(gt_pixels) != len(pred_pixels):
        raise ValueError("Input images must have the same number of pixels.")

    valid = (gt_pixels != 255)
    gt_pixels = gt_pixels[valid]
    pred_pixels = pred_pixels[valid]

    ious = []
    weights = []

    # Calculate IoU separately for every semantic class, then mean them.
    for class_id in range(num_classes):
        if ignore_id == class_id:
            continue

        gt_class = (gt_pixels == class_id)
        pred_class = (pred_pixels == class_id)

        intersection = np.sum(gt_class & pred_class)
        union = np.sum(gt_class | pred_class)

        # If the class is absent from both masks, do not include it (same as iou()).
        if union > 0:
            ious.append(intersection / union)
            weights.append(1 / CLASS_PROPORTIONS[class_id - 1])

    if not ious:
        return np.nan

    ious = np.array(ious)
    weights = np.array(weights)

    weights = weights / np.sum(weights)

    return float(np.sum(ious * weights))


def rand_index(ground_truth, prediction):
    gt_pixels = np.asarray(ground_truth).flatten()
    pred_pixels = np.asarray(prediction).flatten()

    if len(gt_pixels) != len(pred_pixels):
        raise ValueError("Input images must have the same number of pixels.")

    valid = (gt_pixels != 255)
    gt_pixels = gt_pixels[valid]
    pred_pixels = pred_pixels[valid]

    n = len(gt_pixels)
    total_pairs = n * (n - 1) // 2

    # If fewer than two valid pixels remain, RI is undefined for this image.
    if total_pairs == 0:
        return np.nan

    _, counts_gt = np.unique(gt_pixels, return_counts=True)
    _, counts_pred = np.unique(pred_pixels, return_counts=True)

    same_gt = np.sum(counts_gt * (counts_gt - 1) // 2)
    same_pred = np.sum(counts_pred * (counts_pred - 1) // 2)

    pixel_label_pairs = np.stack((gt_pixels, pred_pixels), axis=1)
    _, joint_counts = np.unique(pixel_label_pairs, axis=0, return_counts=True)
    same_both = np.sum(joint_counts * (joint_counts - 1) // 2)

    different_both = total_pairs - same_gt - same_pred + same_both

    return float((same_both + different_both) / total_pairs)