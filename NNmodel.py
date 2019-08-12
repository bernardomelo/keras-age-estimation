from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras import regularizers

def conv_block(inp, filters=32, bn=True, pool=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    return _


input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
_ = conv_block(input_layer, filters=32, bn=False, pool=False)
_ = conv_block(_, filters=32 * 2)
_ = conv_block(_, filters=32 * 3)
_ = conv_block(_, filters=32 * 4)
_ = conv_block(_, filters=32 * 5)
_ = conv_block(_, filters=32 * 6)
bottleneck = GlobalMaxPool2D()(_)

# age calc
_ = Dense(units=128, activation='relu')(bottleneck)
age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

# gender prediction
_ = Dense(units=128, activation='relu')(bottleneck)
gender_output = Dense(units=len(GENDER_ID_MAP), activation='softmax', name='gender_output')(_)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
model.compile(optimizer='rmsprop',
              loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy'},
              loss_weights={'age_output': 2., 'gender_output': 1.},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
# run model.summary()



#####sparse autoencoder#####
# creating autoencoder model
#encoder_inputs = Input(shape=(28, 28, 1))

#conv1 = Conv2D(16, (3, 3), activation='relu', padding="SAME")(encoder_inputs)
#pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
#conv2 = Conv2D(32, (3, 3), activation='relu', padding="SAME")(pool1)
#pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
#flat = Flatten()(pool2)

#enocder_outputs = Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5))(flat)

# upsampling in decoder

#dense_layer_d = Dense(7 * 7 * 32, activation='relu')(enocder_outputs)
#output_from_d = Reshape((7, 7, 32))(dense_layer_d)
#conv1_1 = Conv2D(32, (3, 3), activation='relu', padding="SAME")(output_from_d)
#upsampling_1 = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(conv1_1)
#upsampling_2 = Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2, 2))(upsampling_1)
#decoded_outputs = Conv2DTranspose(1, 3, padding='same', activation='relu')(upsampling_2)

#autoencoder = Model(encoder_inputs, decoded_outputs)

