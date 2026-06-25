cars_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"i love my {c}!",
    lambda c: f"a photo of my dirty {c}.",
    lambda c: f"a photo of my clean {c}.",
    lambda c: f"a photo of my new {c}.",
    lambda c: f"a photo of my old {c}.",
]

cifar10_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a low contrast photo of a {c}.",
    lambda c: f"a high contrast photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a photo of a big {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a low contrast photo of the {c}.",
    lambda c: f"a high contrast photo of the {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the big {c}.",
]

cifar100_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"a low contrast photo of a {c}.",
    lambda c: f"a high contrast photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a photo of a big {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a low contrast photo of the {c}.",
    lambda c: f"a high contrast photo of the {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the big {c}.",
]

dtd_template = [
    lambda c: f"a photo of a {c} texture.",
    lambda c: f"a photo of a {c} pattern.",
    lambda c: f"a photo of a {c} thing.",
    lambda c: f"a photo of a {c} object.",
    lambda c: f"a photo of the {c} texture.",
    lambda c: f"a photo of the {c} pattern.",
    lambda c: f"a photo of the {c} thing.",
    lambda c: f"a photo of the {c} object.",
]

eurosat_template = [
    lambda c: f"a centered satellite photo of {c}.",
    lambda c: f"a centered satellite photo of a {c}.",
    lambda c: f"a centered satellite photo of the {c}.",
]

food101_template = [
    lambda c: f"a photo of {c}, a type of food.",
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

imagenet_template = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]

resisc45_template = [
    lambda c: f"satellite imagery of {c}.",
    lambda c: f"aerial imagery of {c}.",
    lambda c: f"satellite photo of {c}.",
    lambda c: f"aerial photo of {c}.",
    lambda c: f"satellite view of {c}.",
    lambda c: f"aerial view of {c}.",
    lambda c: f"satellite imagery of a {c}.",
    lambda c: f"aerial imagery of a {c}.",
    lambda c: f"satellite photo of a {c}.",
    lambda c: f"aerial photo of a {c}.",
    lambda c: f"satellite view of a {c}.",
    lambda c: f"aerial view of a {c}.",
    lambda c: f"satellite imagery of the {c}.",
    lambda c: f"aerial imagery of the {c}.",
    lambda c: f"satellite photo of the {c}.",
    lambda c: f"aerial photo of the {c}.",
    lambda c: f"satellite view of the {c}.",
    lambda c: f"aerial view of the {c}.",
]

stl10_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

sun397_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

flowers102_template = [
    lambda c: f"a photo of a {c}, a type of flower.",
]

fer2013_template = [
    lambda c: f"a photo of a {c} looking face.",
    lambda c: f"a photo of a face showing the emotion: {c}.",
    lambda c: f"a photo of a face looking {c}.",
    lambda c: f"a face that looks {c}.",
    lambda c: f"they look {c}.",
    lambda c: f"look at how {c} they are.",
]

pcam_template = [
    lambda c: f"this is a photo of {c}",
]

oxfordpets_template = [
    lambda c: f"a photo of a {c}, a type of pet.",
]

sst2_template = [
    lambda c: f"a {c} review of a movie.",
]

emnist_template = [
    lambda c: f'a photo of the digit character: "{c}".',
]

fashionmnist_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

kmnist_template = [
    lambda c: f"a photo of the character {c}.",
]

dataset_to_template = {
    "Cars": cars_template,
    "CIFAR10": cifar10_template,
    "CIFAR100": cifar100_template,
    "DTD": dtd_template,
    "EuroSAT": eurosat_template,
    "Food101": food101_template,
    "GTSRB": gtsrb_template,
    "MNIST": mnist_template,
    "ImageNet": imagenet_template,
    "ImageNet1K": imagenet_template,
    "ImageNet21KP": imagenet_template,
    "RESISC45": resisc45_template,
    "STL10": stl10_template,
    "SUN397": sun397_template,
    "SVHN": svhn_template,
    "Flowers102": flowers102_template,
    "FER2013": fer2013_template,
    "PCAM": pcam_template,
    "OxfordIIITPet": oxfordpets_template,
    "RenderedSST2": sst2_template,
    "EMNIST": emnist_template,
    "FashionMNIST": fashionmnist_template,
    "KMNIST": kmnist_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith("Val"):
        return get_templates(dataset_name.replace("Val", ""))
    assert dataset_name in dataset_to_template, f"Unsupported dataset: {dataset_name}"
    return dataset_to_template[dataset_name]

dataset_descriptions = {
    "Cars": "An image from the Cars dataset, a collection of various car models captured from different angles.",
    "DTD": "An image from the DTD dataset, texture patterns categorized by human descriptions.",
    "EuroSAT": "A centered satellite photo from the EuroSAT dataset, a collection of images showing different land use types such as forests, fields, and highways.",
    "GTSRB": "An image from the GTSRB dataset, real-world road signs captured under different lighting and weather conditions.",
    "MNIST": "An image from the MNIST dataset, grayscale handwritten digits ranging from 0 to 9.",
    "RESISC45": "An aerial image from the RESISC45 dataset, a collection of images representing different land use scenes such as airports, farmlands, and lakes.",
    "SVHN": "An image from the SVHN dataset, house numbers captured from the street.",
    "SUN397": "An image from the SUN397 dataset, a collection of diverse natural and man-made scenes categorized by environment types.",
    "CIFAR100": "An image from the CIFAR-100 dataset, a collection of color images depicting objects and animals grouped into 100 fine-grained classes.",
    "STL10": "An image from the STL-10 dataset, a collection of color images featuring objects and animals for unsupervised learning.",
    "Flowers102": "An image from the Flowers102 dataset, a collection of flower images representing 102 species with varied backgrounds and lighting conditions.",
    "OxfordIIITPet": "An image from the Oxford-IIIT Pet dataset, a collection of 37 cat and dog breeds captured in diverse poses and occlusions.",
    "PCAM": "An image from the PCAM dataset, a collection of pathology images of lymph node tissue samples labeled for tumor detection.",
    "FER2013": "An image from the FER2013 dataset, a collection of grayscale facial expressions labeled with seven emotion categories.",
    "EMNIST": "An image from the EMNIST dataset, a collection of grayscale handwritten characters including digits and letters.",
    "CIFAR10": "An image from the CIFAR-10 dataset, a collection of color images of common objects and animals categorized into ten broad classes.",
    "Food101": "An image from the Food-101 dataset, a collection of 101,000 images depicting dishes from 101 different food categories.",
    "FashionMNIST": "An image from the FashionMNIST dataset, a collection of grayscale images representing clothing items such as shirts and shoes.",
    "RenderedSST2": "An image from the Rendered SST-2 dataset, a collection of text images labeled with sentiment and displayed in different fonts and styles.",
    "KMNIST": "An image from the KMNIST dataset, a collection of grayscale handwritten Japanese characters from Kuzushiji literature.",
}


DATASET_TO_LABEL = {
    "Cars": 0,
    "DTD": 1,
    "EuroSAT": 2,
    "GTSRB": 3,
    "MNIST": 4,
    "RESISC45": 5,
    "SUN397": 6,
    "SVHN": 7,
    # end of 8 tasks
    "CIFAR100": 8,
    "STL10": 9,
    "Flowers102": 10,
    "OxfordIIITPet": 11,
    "PCAM": 12,
    "FER2013": 13,
    # end of 14 tasks
    "EMNIST": 14,
    "CIFAR10": 15,
    "Food101": 16,
    "FashionMNIST": 17,
    "RenderedSST2": 18,
    "KMNIST": 19,
}


def get_dataset_label(dataset):
    assert dataset in DATASET_TO_LABEL, "Error, wrong dataset name"
    return DATASET_TO_LABEL[dataset]

def get_dataset_to_label(datasets):
    return {d: i for i, d in enumerate(datasets)}
