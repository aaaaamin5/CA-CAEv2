import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Add, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import Callback  # [新增] 导入回调函数基类

# ================= [新增功能] 自定义静默进度打印器 =================
class PrintEveryNEpochs(Callback):
    def __init__(self, print_freq, total_epochs):
        super().__init__()
        self.print_freq = print_freq
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # 核心逻辑：每隔 print_freq 轮，或者到达最后一轮时，打印一次 Loss
        if (epoch + 1) % self.print_freq == 0 or (epoch + 1) == self.total_epochs:
            loss = logs.get('loss', 0.0)
            val_loss = logs.get('val_loss', 0.0)
            print(f"      - [训练进度] Epoch {epoch + 1:>3}/{self.total_epochs} | Loss: {loss:.4f} | Val_Loss: {val_loss:.4f}")
# ===================================================================

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(channel // self.ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')

    def call(self, input_feature):
        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, max_pool.shape[1]))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = tf.keras.activations.sigmoid(cbam_feature) 
        cbam_feature = Multiply()([input_feature, cbam_feature])

        return cbam_feature

class CACAE:
    def __init__(
            self,
            input_shape,
            kernel_height1=24,
            kernel_height2=12,
            kernel_height3=6,
            stride1=5,
            stride2=5,
            stride3=4,
            filter1=32,
            filter2=64,
            filter3=128,
            hidden_dim=30,
            optimizer=Adam,
            epochs='auto',          
            batch_size='auto',      
            validation_rate='auto', 
            learning_rate=0.0005
    ):
        # Initialize hyperparameters
        self.kernel_height1 = kernel_height1
        self.kernel_height2 = kernel_height2
        self.kernel_height3 = kernel_height3
        self.stride1 = stride1
        self.stride2 = stride2
        self.stride3 = stride3
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.learning_rate = learning_rate

        # Create a sequential model
        self.model = Sequential()

        # Add convolutional layers to the model
        self.model.add(Conv2D(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu', input_shape=(1, input_shape, 1)))
        self.model.add(Conv2D(filter2, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        self.model.add(ChannelAttention())
        self.model.add(Conv2D(filter3, (1, kernel_height3), strides=(1, stride3), activation='relu', padding='same'))
        self.model.add(ChannelAttention())

        # Flatten the output of the convolutional layers
        self.model.add(Flatten())
        # Add a dense layer for the hidden representation
        self.model.add(Dense(units=hidden_dim, name='HiddenLayer'))
        # Add a dense layer for the output
        self.model.add(Dense(units=filter3 * int(input_shape / (stride3 * stride2 * stride1)), activation='relu'))
        self.model.add(Reshape((1, int(input_shape / (stride3 * stride2 * stride1)), filter3)))

        ## Add deconvolutional layers to the model
        self.model.add(Conv2DTranspose(filter2, (1, kernel_height3), strides=(1, stride3), padding='same', activation='relu'))
        self.model.add(ChannelAttention())
        self.model.add(Conv2DTranspose(filter1, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        self.model.add(ChannelAttention())
        self.model.add(Conv2DTranspose(1, (1, kernel_height1), strides=(1, stride1), activation='relu', padding='same'))

        self.optimizer = optimizer

    def fit(self, data):
        num_samples = data.shape[0]
        
        actual_batch = self.batch_size
        if actual_batch == 'auto':
            if num_samples < 150: actual_batch = 8
            elif num_samples < 400: actual_batch = 16
            elif num_samples < 800: actual_batch = 32
            else: actual_batch = 64
            
        actual_val = self.validation_rate
        if actual_val == 'auto':
            if num_samples < 100: actual_val = 0.1
            elif num_samples < 300: actual_val = 0.15
            else: actual_val = 0.2
            
        actual_eps = self.epochs
        if actual_eps == 'auto':
            if num_samples < 150: actual_eps = 200
            elif num_samples < 400: actual_eps = 300
            else: actual_eps = 500
            
        print(f"  👉 [模型自适应] 样本数: {num_samples} | BatchSize: {actual_batch}, Epochs: {actual_eps}, ValRate: {actual_val}")

        # Compile the model
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer(learning_rate=self.learning_rate))
        
        # Reshape the input data
        data_train = tf.reshape(data, [-1, 1, data.shape[1], 1])
        
        # ================= [调用新增的回调] =================
        # 设置打印频率为 100 轮
        my_logger = PrintEveryNEpochs(print_freq=100, total_epochs=actual_eps)
        
        # Train the model (加入 callbacks 列表)
        self.model.fit(data_train, data_train,
                       epochs=actual_eps,
                       batch_size=actual_batch,
                       validation_split=actual_val,
                       verbose=0,  # 依然屏蔽默认的烦人进度条
                       callbacks=[my_logger]) 
        # ====================================================

    def extract_feature(self, x):
    # f = tf.keras.backend.function(self.model.input, self.model.get_layer('HiddenLayer').output)
    # Sequential 使用 inputs[0]
        feature_model = tf.keras.Model(inputs=self.model.inputs[0], outputs=self.model.get_layer('HiddenLayer').output)
        x = tf.reshape(x, [-1, 1, x.shape[1], 1])
        return feature_model.predict(x, verbose=0)
