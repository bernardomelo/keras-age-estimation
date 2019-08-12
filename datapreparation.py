from keras.utils import to_categorical
from PIL import Image


def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, genders = [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, gender = r['file'], r['age'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IM_WIDTH, IM_HEIGHT))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(genders)]
                images, ages, genders = [], [], []
        if not for_training:
            break

    yield np.array(images), [np.array(ages), np.array(genders)]