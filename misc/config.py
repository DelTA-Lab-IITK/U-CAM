############ These paths need to be changed accordingly #####################
qa_path = 'vqa/'  # directory containing the question and annotation jsons
train_path = 'VQA/Images/mscoco/train2014'  # directory of training images
val_path = 'VQA/Images/mscoco/val2014'  # directory of validation images
test_path = 'VQA/Images/mscoco/test2015'  # directory of test images
preprocessed_path = 'resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to


task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 10
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 100
batch_size = 220
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000

monte_carlo_simulations=100
ans_classes=max_answers
