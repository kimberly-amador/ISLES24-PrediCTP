import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Reshape, Embedding, Conv2DTranspose, Concatenate, Activation, MultiHeadAttention, LayerNormalization, GlobalMaxPooling1D


# -----------------------------------
#        MODEL ARCHITECTURE
# -----------------------------------

class ENCODER:
    @staticmethod
    def build(reg=l2(), shape=(416, 416), init='he_normal'):
        # Create the model
        i = Input(shape=(*shape, 1))

        # The first two layers will learn a total of 16 filters with a 3x3x3 kernel size
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(i)
        d1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d1')(o)
        o = Conv2D(16, (2, 2), strides=(2, 2))(d1)  # down-sampling shape (208, 208)

        # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d2')(o)
        o = Conv2D(32, (2, 2), strides=(2, 2))(d2)  # down-sampling shape (104, 104)

        # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d3')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d3)  # down-sampling shape (52, 52)

        # Could add one more conv block here
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d4')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d4)  # down-sampling shape (26, 26)

        # Encoder outputs
        size = o.shape[1] * o.shape[2]
        output = Reshape((1, size))(o)
        down = [d1, d2, d3, d4]

        return Model(inputs=i, outputs=[output, *down], name='Encoder')


class DECODER:
    @staticmethod
    def build(inputTensor, down, n_classes, reg=l2(), init='he_normal'):
        # Reshape to fit the decoder
        o = Reshape((26, 26, 1))(inputTensor)  # TODO: automate reshape size

        # Could add one more conv block here
        u4 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Up-sampling
        concat = Concatenate()([u4, down[3]])
        # Stack two more layers, keeping the size of each filter as 3x3x3 but decreasing to 64 total learned filters
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u3 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Up-sampling
        concat = Concatenate()([u3, down[2]])
        # Stack two more layers, keeping the size of each filter as 3x3x3 but decreasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2))(o)  # Up-sampling
        concat = Concatenate()([u2, down[1]])
        # Stack two more layers, keeping the size of each filter as 3x3x3 but decreasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2))(o)  # Up-sampling
        concat = Concatenate()([u1, down[0]])
        # Stack three more layers, two 16 filter layers and a softmax layer
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        output = Conv2D(n_classes, 1, padding="same", name='Logit')(o)
        output = Activation('softmax')(output)

        return output


class SelfAttention:
    @staticmethod
    def build(inputTensor, n_layers, n_heads, dropout, positional_encoding=True):
        # Define embed_dim
        embed_dim = inputTensor.shape[2]

        # Positional embedding layer (optional)
        if positional_encoding:
            sequence_length = inputTensor.shape[1]  # inputs are of shape: (batch_size, n_timepoints, embed_dim)
            positions = tf.range(start=0, limit=sequence_length, delta=1)
            embedded_positions = Embedding(input_dim=sequence_length, output_dim=embed_dim)(positions)
            inputTensor += embedded_positions

        # Multi-head self-attention layer
        for _ in range(n_layers):
            attention_output, attention_scores = MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim, dropout=dropout)(inputTensor, inputTensor, return_attention_scores=True)
            inputTensor = LayerNormalization()(inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs

        return inputTensor


class MultimodalFusion:
    @staticmethod
    def build(image_embed, clinical_embed, n_heads=8, dropout=0.2):
        # Define embed_dim
        embed_dim = image_embed.shape[2]

        # Co-attention: Query - Imaging; Key & Value - Metadata
        A_attention_output, A_attention_scores = MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim, dropout=dropout)(image_embed, clinical_embed, return_attention_scores=True)
        A_proj_input = LayerNormalization()(image_embed + A_attention_output)
        A_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dense(embed_dim), ])(A_proj_input)
        A_x = LayerNormalization()(A_proj_input + A_proj_output)  # shape (None, 32, 2704)

        # Reduce output sequence through pooling layer
        A_x = GlobalMaxPooling1D()(A_x)  # shape (None, 2704)
        A_output = Dropout(dropout)(A_x)

        return A_output


class TRANSFORMER:
    @staticmethod
    def build(inputTensor, n_layers, n_heads, dropout, positional_encoding=True):
        # Define embed_dim
        embed_dim = inputTensor.shape[2]

        # Positional embedding layer (optional)
        if positional_encoding:
            sequence_length = inputTensor.shape[1]  # inputs are of shape: (batch_size, n_timepoints, embed_dim)
            positions = tf.range(start=0, limit=sequence_length, delta=1)
            embedded_positions = Embedding(input_dim=sequence_length, output_dim=embed_dim)(positions)
            inputTensor += embedded_positions

        # Multi-head self-attention layer
        for _ in range(n_layers):
            attention_output, attention_scores = MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim, dropout=dropout)(inputTensor, inputTensor, return_attention_scores=True)
            proj_input = LayerNormalization()(inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs
            proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dense(embed_dim), ])(proj_input)
            inputTensor = LayerNormalization()(proj_input + proj_output)

        # Reduce output sequence through pooling layer
        x = GlobalMaxPooling1D()(inputTensor)
        output = Dropout(dropout)(x)

        return output
