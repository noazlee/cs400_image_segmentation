#A link to a broader description of the Rand Index, as well as a simple example,
#can be found at this link: 

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def RandIndex(labels1, labels2):

    #flatten any inpuit into a 1D array
    pixels1 = np.asarray(labels1).flatten()
    pixels2 = np.asarray(labels2).flatten()

    #comparing pairs of pixels is ineffective if the two images have different numbers of pixels
    if len(pixels1) != len(pixels2):
        raise ValueError("Input images must have the same number of pixels.")

    #The AI4Mars dataset uses 255 to represent pixels that are unlabeled or NULL.
    valid = (pixels1 != 255) & (pixels2 != 255)
    #ignore those pixels
    pixels1 = pixels1[valid]
    pixels2 = pixels2[valid]

    #First, we calculate the total number of pixel pairs in the images segmentations, which is N choose 2 = N*(N-1)/2.
    total_pairs = len(pixels1) * (len(pixels1) - 1) // 2

    #For the two segmentations, np.unique returns an array of the unique pixel values (there should be one 
    #for each segmentation class) that appear in the segmentation, and an array of the counts of how many times 
    #each unique value appears. We only care about the array of counts, so we use _ to ignore the unique pixel values.
    _, counts1 = np.unique(pixels1, return_counts=True)
    _, counts2 = np.unique(pixels2, return_counts=True)

    #For each segmentation, we now calculate the total number of pixel pairs that have the same label in that segmentation. 
    #We do this by finding the number of pairs in each class (which is the array of counts) by doing NC2, and then 
    #summing over these values
    same1 = np.sum(counts1 * (counts1 - 1) // 2)
    same2 = np.sum(counts2 * (counts2 - 1) // 2)

    #Now we need to calculate the number of pixel pairs that have the same label in both segmentations.
    #We do this by stacking the two arrays of pixel values into a 2D array, where each row is a pair of pixel values from the two segmentations. 
    #We then use np.unique to find the unique pairs of pixel values and their counts, and then calculate the number of pairs that 
    #have the same label in both segmentations by doing the same thing we did for the individual segmentations above.
    pairs = np.stack((pixels1, pixels2), axis=1) 
    _, joint_counts = np.unique(pairs, axis=0, return_counts=True)
    same_both = np.sum(joint_counts * (joint_counts - 1) // 2)

    #Finally, we can calculate the number of pixel pairs that have different labels in both segmentations.
    #We do this by subtracting the number of pairs that have the same label in either segmentation from the total number of pairs.
    #We add back on the pairs that have the same label in both segmentations, since we subtracted those pairs twice.
    different_both = (total_pairs - same1 - same2 + same_both)

    return (same_both + different_both) / total_pairs


#ground_truth = np.array(Image.open("AI4Mars_label.png").convert("L"))
#prediction = model_output.argmax(axis=0)
#score = RandIndex(ground_truth, prediction)
