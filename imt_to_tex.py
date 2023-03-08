import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2
from skimage.measure import regionprops, label
from tqdm import tqdm


def extract_features(image, if_symb_is_i):
    if image.ndim == 3:
        gray = np.mean(image, 2)
        gray[gray > 0] = 1
        labeled = label(gray)
    else:
        labeled = image.astype("uint8")
    props = regionprops(labeled)[0]
    extants = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    rr, cc = props.centroid_local
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    feret = (props.feret_diameter_max - 1) / np.max(props.image.shape)
    return np.array([extants, eccentricity, euler, rr, cc, feret, if_symb_is_i], dtype="f4") 


def image2text(image) -> str:
    gray = np.mean(image,2)
    gray[gray > 0] = 1
    labeled = label(gray)

    sort_letters = sorted(regionprops(labeled),key=lambda r: r.bbox[1])
    slet_len = len(sort_letters)

    spaces, letter_i, coord_i = [], [], []

    for i in range(1, slet_len - 1):
        if abs(sort_letters[i].bbox[1] - sort_letters[i + 1].bbox[1]) < 10 or abs(sort_letters[i].bbox[1] - sort_letters[i - 1].bbox[1]) < 10:
            letter_i.append(sort_letters[i].bbox)
            coord_i.append(i)
        if i < slet_len - 1 and sort_letters[i + 1].bbox[1] - sort_letters[i].bbox[3] > 20:
            spaces.append(i)
            
    for i in range(0, len(letter_i), 2):
        lbn = labeled[letter_i[i + 1][0]:letter_i[i + 1][2], letter_i[i + 1][1]:letter_i[i + 1][3]]
        lbn[lbn > 0] = max(labeled[letter_i[i][0]:letter_i[i][2], letter_i[i][1]:letter_i[i][3]][3])
        labeled[letter_i[i + 1][0]:letter_i[i + 1][2], letter_i[i + 1][1]:letter_i[i + 1][3]] = lbn
        for j in range(len(coord_i)):
            if j >= i + 2:
                coord_i[j] -= 1
                spaces[j - 2] -= 1


    sort_letters = sorted(regionprops(labeled),key=lambda r: r.bbox[1])
    answer = []
    coord_i = coord_i[::2]
    
    for region in sort_letters:
        if sort_letters.index(region) in coord_i:
            features = extract_features(region.image, True).reshape(1, -1)
        else:
            features = extract_features(region.image, False).reshape(1, -1)
        ret, _, _, _ = knn.findNearest(features, 2)
        answer.append(class2symb[int(ret)])

    answer = "".join(answer)
    
    for j in spaces:
        answer = answer[:j + 1] + " " + answer[j + 1:]
    
    return  answer


if __name__ == "__main__":
    text_images = [plt.imread(path) for path in pathlib.Path("./out/").glob("*.png")]
    train_images = {}

    for path in tqdm(sorted(pathlib.Path("./out/train").glob("*"))):
        train_images[path.name[-1]] = [plt.imread(image_path) for image_path in sorted(path.glob("*.png"))]

    knn = cv2.ml.KNearest_create()
    train, responses = [], []
    symb2class = {symbol: i for i, symbol in enumerate(train_images)}
    class2symb = {value: key for key, value in symb2class.items()}

    for symbol in tqdm(train_images):
        for image in train_images[symbol]:
            train.append(extract_features(image, symbol == "i"))
            responses.append(symb2class[symbol])

    knn.train(np.array(train, dtype="f4"), cv2.ml.ROW_SAMPLE, np.array(responses))

    number_of_images = 3

    for k in range(number_of_images):
        answer = image2text(text_images[k])
        print(f"Image number {k+1} answer: {answer}.")