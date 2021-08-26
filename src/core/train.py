import tensorflow as tf
import pickle
import tensorflow_hub as hub

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    mobilenet_v2,
    input_shape=(224, 224, 3),
    trainable=False)

classifier = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(2)
    ])


from pathlib import Path

dir_ = Path('/content/drive/MyDrive/zero/fruits/')
file_paths = list(dir_.glob(r'**/*.jpeg'))

import pandas as pd
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("/")[-2].capitalize() \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    
    return df

df = proc_img(file_paths)

print(f'Number of pictures in the dataset: {df.shape[0]}\n')
print(f'Number of different labels: {len(df.Label.unique())}\n')
print(f'Labels: {df.Label.unique()}')

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.1
)

train_images = train_generator.flow_from_dataframe(
    dataframe=df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    #batch_size=5,
    shuffle=True,
    seed=0,
    subset='training',
#         rotation_range=30, # Uncomment to use data augmentation
#         zoom_range=0.15,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.15,
#         horizontal_flip=True,
#         fill_mode="nearest"
)

classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

history = classifier.fit(train_images, epochs=2)

export_path = "/saved_models/{}".format(int(1))
classifier.save(export_path)

classes = list(train_images.class_indices.keys())
with open("utils/class_labels.pickle","wb") as file:
    pickle.dump(classes, file)
