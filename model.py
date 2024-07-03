from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from functions import DC_layer, DHC_MHA, AMG_MHA

class LungConVTConfig:
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.n_classes = 1000
        self.initial_filters = 32
        self.combo_filters = (32, 64, 128, 256)
        self.transformer_params = {
            'transformer_1': {'num_heads': [4, 8], 'ff_dim': 128, 'dropout_rate': 0.2},
            'transformer_2': {'num_heads': [2, 2], 'ff_dim': [64, 128], 'depth': [4, 8]}
        }

class LungConVT:
    def __init__(self, config: LungConVTConfig):
        self.config = config

    def build_model(self):
        config = self.config
        
        # Define the input layer with the specified shape
        input_tensor = Input(shape=config.input_shape)

        # Initial convolutional layer with strides, followed by batch normalization and ReLU activation
        x = Conv2D(filters=config.initial_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second convolutional layer
        x = Conv2D(filters=config.initial_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)

        # DC layers for feature extraction and downsampling
        x = DC_layer(x, filters=config.combo_filters[0], strides=(1, 1))
        x = DC_layer(x, filters=config.combo_filters[1], strides=(2, 2))

        # Additional DC layers to refine features
        x = DC_layer(x, filters=config.combo_filters[1], strides=(1, 1))
        x = DC_layer(x, filters=config.combo_filters[2], strides=(2, 2))

        # First transformer block to capture global context with multi-head self-attention
        transformer_1 = config.transformer_params['transformer_1']
        x = DHC_MHA(x, num_heads=transformer_1['num_heads'][0], ff_dim=transformer_1['ff_dim'], dropout_rate=transformer_1['dropout_rate'])
        x = DC_layer(x, filters=config.combo_filters[2], strides=(2, 2))

        # Second transformer block with increased number of attention heads
        x = DHC_MHA(x, num_heads=transformer_1['num_heads'][1], ff_dim=transformer_1['ff_dim'], dropout_rate=transformer_1['dropout_rate'])

        # Output shape of the last transformer block
        print("Last Transformer_Shape", x.shape)

        # Calculate the number of patches for non-overlapping patch embedding
        num_patches = int((x.shape[1] * x.shape[2]) / 4)
        print("Patch Size", num_patches)

        # Reshape the tensor to have non-overlapping patches
        x = Reshape((4, num_patches, config.combo_filters[2]))(x)
        print("non_overlapping_patches Size", x.shape)

        # Further transformer blocks to process the reshaped patches
        transformer_2 = config.transformer_params['transformer_2']
        x = AMG_MHA(x, num_heads=transformer_2['num_heads'][0], ff_dim=transformer_2['ff_dim'][0], depth=transformer_2['depth'][0])
        x = DC_layer(x, filters=config.combo_filters[2], strides=(1, 1))
        x = AMG_MHA(x, num_heads=transformer_2['num_heads'][1], ff_dim=transformer_2['ff_dim'][1], depth=transformer_2['depth'][1])

        # Final DC layer to refine the features before classification
        x = DC_layer(x, filters=config.combo_filters[3], strides=(1, 1))

        # Global average pooling to reduce the spatial dimensions
        x = GlobalAveragePooling2D()(x)

        # Output dense layer with softmax activation for classification
        output_tensor = Dense(units=config.n_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_tensor, outputs=output_tensor)

        return model

# Usage
config = LungConVTConfig()
lung_con_vt = LungConVT(config)
model = lung_con_vt.build_model()
