# R4.04 Optimisation — Neural Networks

## Description

This Jupyter notebook implements a neural network to predict the "Benchmark (MMLU)" scores of various large language models (LLMs) based on their characteristics. The notebook follows these key steps, based on the three aforementioned tutorials:

1. Feature Engineering: Separate features (X) from the target variable (Y), which is the MMLU benchmark score

2. Categorical Encoding: Use of LabelEncoder to convert categorical columns ('Model', 'Provider', 'Open-Source') into numerical values

3. Data Splitting: Split the data between a training set and a testing set (80/20)

4. Feature Standardization: Apply standard scaling to normalize the feature values

5. Neural Network Architecture, i.e. a sequential model with:
    - An input layer connected to a hidden layer of 64 neurons with ReLU activation
    - A second hidden layer with 32 neurons and ReLU activation
    - An output layer with a single neuron and linear activation (for regression)

6. Model Training: Compiling and training of the model

7. Evaluation: Calculates Mean Absolute Error (MAE) and visualize actual vs. predicted values

8. Hyperparameter Tuning: Experiments with different network configurations, varying the number of layers (1-3) and epochs (50-500) to find the optimal model architecture

## Dataset description

Dataset and description taken from [https://www.kaggle.com/datasets/samayashar/large-language-models-comparison-dataset](https://www.kaggle.com/datasets/samayashar/large-language-models-comparison-dataset)


### About Dataset

This dataset provides a comparison of various Large Language Models (LLMs) based on their performance, cost, and efficiency. It includes important details like speed, latency, benchmarks, and pricing, helping users understand how different models stack up against each other.

### Key Details:

File Name: llm_comparison_dataset.csv
Size: 14.57 kB
Total Columns: 15
License: CC0 (Public Domain)

### What’s Inside?

Here are some of the key metrics included in the dataset:

*Context Window*: Maximum number of tokens the model can process at once.
*Speed (tokens/sec)*: How fast the model generates responses.
*Latency (sec)*: Time delay before the model responds.
*Benchmark Scores*: Performance ratings from MMLU (academic tasks) and Chatbot Arena (real-world chatbot performance).
*Open-Source*: Indicates if the model is publicly available or proprietary.
*Price per Million Tokens*: The cost of using the model for one million tokens.
*Training Dataset Size*: Amount of data used to train the model.
*Compute Power*: Resources needed to run the model.
*Energy Efficiency*: How much power the model consumes.


```python
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
data = pd.read_csv('llm_comparison_dataset.csv')
# Online version
# data = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/6732641/10841276/llm_comparison_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250320%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250320T152312Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1120e5d1d8e2427437a6c850a6028455404f652ecb23897246731c337c0a98e9d2791ccc9a9263ea2b82bd299f3db69925f1411ebecea8c219d9b611e4bb5be016c331c908ebe9a287099bcd3f23ef04c162d70580589f9dec553399d0af0a079adad8356d39133cf1920fe374d8a4ab766428554e3faaa4efdb43cbbcf39bb4c57de35e0e7335d45d5cdcf36451b8a686753cd94bbfcf30fc04ccb2f8e108f1fe3350c5e88fd44fac879b68c8ba43c07e82e9c02cb5ab7ae4304cd650d9ad776d650bebe352256d27dcc8e15c9df32b5ea47231c7435cd67d4c46cf127fc02478cdd67f703bb276811655aa9f4637314c6ae6f573cdd8a0a5bfe75c71e12a50')
print(data.head())
```

            Model  Provider  Context Window  Speed (tokens/sec)  Latency (sec)  \
    0  DeepSeek-4  Deepseek          128000                  95           2.74   
    1     Llama-8   Meta AI          300000                 284           3.21   
    2     Llama-5   Meta AI          300000                 225           2.95   
    3  DeepSeek-3  Deepseek         2000000                 242          12.89   
    4  DeepSeek-8  Deepseek         1000000                  71           3.80   
    
       Benchmark (MMLU)  Benchmark (Chatbot Arena)  Open-Source  \
    0                85                       1143            1   
    1                71                       1390            1   
    2                85                       1406            0   
    3                72                       1264            1   
    4                77                       1381            1   
    
       Price / Million Tokens  Training Dataset Size  Compute Power  \
    0                   18.81              760952565             13   
    1                    3.98               22891342             22   
    2                    1.02              827422145             21   
    3                   27.63              694305632             86   
    4                   18.52              378552278             92   
    
       Energy Efficiency  Quality Rating  Speed Rating  Price Rating  
    0               0.50               2             2             3  
    1               2.07               1             3             3  
    2               0.95               2             3             2  
    3               3.51               1             3             3  
    4               1.80               2             2             3  



```python
# Define features (X) and target (Y)
X = data.drop(columns=['Benchmark (MMLU)'])  # Drop the MMLU column
Y = data['Benchmark (MMLU)']  # Target variable

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
```

    Shape of X: (200, 14)
    Shape of Y: (200,)



```python
# Encode categorical columns
label_encoder = LabelEncoder()

categorical_columns = ['Model', 'Provider', 'Open-Source']
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

print(X.head())
```

       Model  Provider  Context Window  Speed (tokens/sec)  Latency (sec)  \
    0     20         3          128000                  95           2.74   
    1     51         5          300000                 284           3.21   
    2     48         5          300000                 225           2.95   
    3     19         3         2000000                 242          12.89   
    4     24         3         1000000                  71           3.80   
    
       Benchmark (Chatbot Arena)  Open-Source  Price / Million Tokens  \
    0                       1143            1                   18.81   
    1                       1390            1                    3.98   
    2                       1406            0                    1.02   
    3                       1264            1                   27.63   
    4                       1381            1                   18.52   
    
       Training Dataset Size  Compute Power  Energy Efficiency  Quality Rating  \
    0              760952565             13               0.50               2   
    1               22891342             22               2.07               1   
    2              827422145             21               0.95               2   
    3              694305632             86               3.51               1   
    4              378552278             92               1.80               2   
    
       Speed Rating  Price Rating  
    0             2             3  
    1             3             3  
    2             3             2  
    3             3             3  
    4             2             3  



```python
# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
```

    X_train shape: (160, 14)
    X_test shape: (40, 14)



```python
# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])
```

    [[ 0.68263172  0.54806646 -0.54833634  0.25191367 -1.11334056  1.33384712
       1.          1.74365182  1.24569421 -1.49975851  1.01963639 -1.09755466
      -0.4100205   0.30030045]
     [ 1.59746622 -1.61180137 -0.54833634 -0.42690041  1.56234433 -1.44795354
       1.          0.69178481  0.24677989 -1.36293094  1.14982643  0.16581761
      -0.4100205   0.30030045]
     [-0.99456487 -0.74785424  1.97621112  0.63614805 -0.03294923  1.15267943
      -1.         -1.60519649 -0.79579965 -0.47355169 -1.47453066  1.42918988
       1.1900595  -2.90290436]
     [ 0.12356619  0.1160929   1.97621112  1.16126838  0.55783333 -1.08561816
      -1.         -0.62738833  1.56962476 -0.26831033  1.56780497 -1.09755466
       1.1900595   0.30030045]
     [-1.70610282 -1.1798278  -0.73362423 -0.32443791  1.03118204 -0.58886805
       1.          1.51221793 -1.09551508 -0.54196548 -1.43341801  0.16581761
      -0.4100205   0.30030045]]



```python
# Initialize the neural network
model = Sequential()
```


```python
# Add the input layer and first hidden layer
input_dim = X_train.shape[1]
model.add(Dense(units=64, activation='relu', input_dim=input_dim))
```

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)



```python
# Add a second hidden layer
model.add(Dense(units=32, activation='relu'))
```


```python
# Add the output layer
model.add(Dense(units=1, activation='linear'))  # Linear activation for regression
```


```python
# Compile the ANN
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```


```python
# Train the ANN
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=32)
```

    Epoch 1/100


    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 6085.3682 - mae: 77.3842 - val_loss: 6383.1958 - val_mae: 79.3403
    Epoch 2/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 6105.0898 - mae: 77.4559 - val_loss: 6333.6484 - val_mae: 79.0302
    Epoch 3/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 6070.5186 - mae: 77.1999 - val_loss: 6284.0332 - val_mae: 78.7180
    Epoch 4/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5959.0249 - mae: 76.5394 - val_loss: 6232.5488 - val_mae: 78.3931
    Epoch 5/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5881.5044 - mae: 76.0583 - val_loss: 6178.2280 - val_mae: 78.0496
    Epoch 6/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5843.2065 - mae: 75.7902 - val_loss: 6121.0186 - val_mae: 77.6866
    Epoch 7/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5801.3779 - mae: 75.5410 - val_loss: 6060.1826 - val_mae: 77.2986
    Epoch 8/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 5682.9678 - mae: 74.7194 - val_loss: 5995.1953 - val_mae: 76.8820
    Epoch 9/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5765.4961 - mae: 75.2804 - val_loss: 5925.1963 - val_mae: 76.4308
    Epoch 10/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5729.6704 - mae: 75.0570 - val_loss: 5850.0991 - val_mae: 75.9436
    Epoch 11/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5635.7886 - mae: 74.4578 - val_loss: 5769.5723 - val_mae: 75.4178
    Epoch 12/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 5482.3521 - mae: 73.3738 - val_loss: 5683.3110 - val_mae: 74.8502
    Epoch 13/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5327.3228 - mae: 72.3317 - val_loss: 5590.6387 - val_mae: 74.2356
    Epoch 14/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5341.9102 - mae: 72.4483 - val_loss: 5491.4150 - val_mae: 73.5720
    Epoch 15/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5175.3354 - mae: 71.2799 - val_loss: 5385.0352 - val_mae: 72.8536
    Epoch 16/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 5075.0469 - mae: 70.6016 - val_loss: 5272.0571 - val_mae: 72.0822
    Epoch 17/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4830.9858 - mae: 68.9383 - val_loss: 5151.4443 - val_mae: 71.2488
    Epoch 18/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4772.8652 - mae: 68.4294 - val_loss: 5022.4800 - val_mae: 70.3471
    Epoch 19/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4758.5425 - mae: 68.3588 - val_loss: 4886.3184 - val_mae: 69.3822
    Epoch 20/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4535.9990 - mae: 66.7557 - val_loss: 4743.0469 - val_mae: 68.3510
    Epoch 21/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4407.1772 - mae: 65.7609 - val_loss: 4591.9941 - val_mae: 67.2462
    Epoch 22/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 4220.8979 - mae: 64.4395 - val_loss: 4433.1577 - val_mae: 66.0642
    Epoch 23/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 4131.4155 - mae: 63.6688 - val_loss: 4265.6802 - val_mae: 64.7923
    Epoch 24/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 4007.1692 - mae: 62.6885 - val_loss: 4091.5520 - val_mae: 63.4423
    Epoch 25/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 3734.4805 - mae: 60.5049 - val_loss: 3912.2705 - val_mae: 62.0180
    Epoch 26/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 3499.7483 - mae: 58.4937 - val_loss: 3726.5110 - val_mae: 60.5051
    Epoch 27/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 3377.4023 - mae: 57.5052 - val_loss: 3535.7056 - val_mae: 58.9070
    Epoch 28/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 3169.8386 - mae: 55.7299 - val_loss: 3339.6831 - val_mae: 57.2157
    Epoch 29/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 2881.4509 - mae: 53.0211 - val_loss: 3141.7375 - val_mae: 55.4518
    Epoch 30/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 2666.9280 - mae: 51.0327 - val_loss: 2941.0522 - val_mae: 53.5975
    Epoch 31/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 2561.8506 - mae: 49.9182 - val_loss: 2738.6680 - val_mae: 51.6569
    Epoch 32/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 2344.2446 - mae: 47.6914 - val_loss: 2536.3569 - val_mae: 49.6339
    Epoch 33/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 2151.6331 - mae: 45.6338 - val_loss: 2336.1301 - val_mae: 47.5386
    Epoch 34/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 1842.4133 - mae: 42.1220 - val_loss: 2141.8257 - val_mae: 45.4045
    Epoch 35/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 1718.5857 - mae: 40.6272 - val_loss: 1952.0369 - val_mae: 43.2074
    Epoch 36/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 1526.3528 - mae: 38.0850 - val_loss: 1766.9736 - val_mae: 40.9446
    Epoch 37/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 1363.3346 - mae: 35.7504 - val_loss: 1590.7490 - val_mae: 38.6510
    Epoch 38/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 1219.4191 - mae: 33.7689 - val_loss: 1422.8639 - val_mae: 36.3256
    Epoch 39/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 1051.6862 - mae: 31.3036 - val_loss: 1265.4221 - val_mae: 33.9872
    Epoch 40/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 913.3768 - mae: 29.0162 - val_loss: 1119.9641 - val_mae: 31.6618
    Epoch 41/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 773.7203 - mae: 26.2740 - val_loss: 987.2826 - val_mae: 29.3690
    Epoch 42/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 663.1933 - mae: 24.0121 - val_loss: 867.6874 - val_mae: 27.1232
    Epoch 43/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 558.9672 - mae: 22.0177 - val_loss: 759.9659 - val_mae: 24.9255
    Epoch 44/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 441.4594 - mae: 19.3403 - val_loss: 665.4503 - val_mae: 22.8149
    Epoch 45/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 373.0986 - mae: 17.6401 - val_loss: 583.3384 - val_mae: 20.8268
    Epoch 46/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 320.0217 - mae: 16.0562 - val_loss: 512.9631 - val_mae: 19.1982
    Epoch 47/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 272.3735 - mae: 14.6582 - val_loss: 452.9390 - val_mae: 17.8419
    Epoch 48/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 225.8789 - mae: 13.2780 - val_loss: 402.8890 - val_mae: 16.6408
    Epoch 49/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 189.9443 - mae: 11.9305 - val_loss: 361.0868 - val_mae: 15.5323
    Epoch 50/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 166.1404 - mae: 10.7630 - val_loss: 326.2974 - val_mae: 14.6812
    Epoch 51/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 146.0309 - mae: 9.9256 - val_loss: 297.9104 - val_mae: 14.1168
    Epoch 52/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 133.3428 - mae: 9.4914 - val_loss: 274.0958 - val_mae: 13.6155
    Epoch 53/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 114.9680 - mae: 8.4681 - val_loss: 255.0356 - val_mae: 13.2096
    Epoch 54/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 118.2992 - mae: 8.6466 - val_loss: 238.8860 - val_mae: 12.8618
    Epoch 55/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 117.6899 - mae: 8.6161 - val_loss: 225.4409 - val_mae: 12.5335
    Epoch 56/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 109.8046 - mae: 7.9842 - val_loss: 214.8241 - val_mae: 12.2500
    Epoch 57/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 86.4524 - mae: 7.2519 - val_loss: 205.7370 - val_mae: 11.9918
    Epoch 58/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 101.0834 - mae: 7.6799 - val_loss: 198.2835 - val_mae: 11.7842
    Epoch 59/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 102.7846 - mae: 7.7747 - val_loss: 191.5122 - val_mae: 11.5834
    Epoch 60/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 87.1696 - mae: 6.9806 - val_loss: 185.7514 - val_mae: 11.3996
    Epoch 61/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 99.1117 - mae: 7.7580 - val_loss: 181.2076 - val_mae: 11.2436
    Epoch 62/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 100.2699 - mae: 7.7456 - val_loss: 177.1301 - val_mae: 11.1013
    Epoch 63/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 83.8320 - mae: 6.8459 - val_loss: 173.6222 - val_mae: 10.9808
    Epoch 64/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 80.6609 - mae: 6.9575 - val_loss: 170.6964 - val_mae: 10.8796
    Epoch 65/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 87.6550 - mae: 7.1540 - val_loss: 167.8262 - val_mae: 10.7784
    Epoch 66/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 71.3341 - mae: 6.4974 - val_loss: 165.1988 - val_mae: 10.6843
    Epoch 67/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 78.6848 - mae: 6.8176 - val_loss: 162.8529 - val_mae: 10.5964
    Epoch 68/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 80.5031 - mae: 7.1256 - val_loss: 160.9034 - val_mae: 10.5197
    Epoch 69/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 76.1741 - mae: 6.8671 - val_loss: 159.2095 - val_mae: 10.4550
    Epoch 70/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 72.6287 - mae: 6.8370 - val_loss: 157.3477 - val_mae: 10.3845
    Epoch 71/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 71.5290 - mae: 6.6581 - val_loss: 155.7181 - val_mae: 10.3213
    Epoch 72/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 64.8302 - mae: 6.5248 - val_loss: 154.1260 - val_mae: 10.2601
    Epoch 73/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 68.4066 - mae: 6.4032 - val_loss: 152.6069 - val_mae: 10.2008
    Epoch 74/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 69.4723 - mae: 6.4777 - val_loss: 151.2680 - val_mae: 10.1533
    Epoch 75/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 58.5628 - mae: 6.0148 - val_loss: 149.8974 - val_mae: 10.0986
    Epoch 76/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 74.5735 - mae: 6.5477 - val_loss: 149.0406 - val_mae: 10.0618
    Epoch 77/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 60.7491 - mae: 6.0979 - val_loss: 147.9509 - val_mae: 10.0157
    Epoch 78/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 66.5741 - mae: 6.3235 - val_loss: 146.7869 - val_mae: 9.9694
    Epoch 79/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 73.2043 - mae: 6.4914 - val_loss: 145.8363 - val_mae: 9.9273
    Epoch 80/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 59.2704 - mae: 6.1066 - val_loss: 144.8488 - val_mae: 9.8798
    Epoch 81/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 62.5935 - mae: 6.2495 - val_loss: 143.9681 - val_mae: 9.8396
    Epoch 82/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 54.1277 - mae: 5.7242 - val_loss: 143.2194 - val_mae: 9.8067
    Epoch 83/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 64.5592 - mae: 6.3926 - val_loss: 142.3459 - val_mae: 9.7733
    Epoch 84/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 56.0801 - mae: 5.7183 - val_loss: 141.7579 - val_mae: 9.7398
    Epoch 85/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 56.2003 - mae: 5.8548 - val_loss: 140.7332 - val_mae: 9.7000
    Epoch 86/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 62.7799 - mae: 6.0073 - val_loss: 139.8829 - val_mae: 9.6619
    Epoch 87/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 58.3778 - mae: 6.0896 - val_loss: 139.4536 - val_mae: 9.6390
    Epoch 88/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 55.3097 - mae: 5.7785 - val_loss: 138.8862 - val_mae: 9.6135
    Epoch 89/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 56.2785 - mae: 5.8199 - val_loss: 138.1145 - val_mae: 9.5792
    Epoch 90/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 55.9373 - mae: 5.6212 - val_loss: 137.7243 - val_mae: 9.5554
    Epoch 91/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 56.0920 - mae: 5.8073 - val_loss: 137.1549 - val_mae: 9.5356
    Epoch 92/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 60.6452 - mae: 6.0356 - val_loss: 136.4670 - val_mae: 9.5089
    Epoch 93/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 56.6528 - mae: 5.8599 - val_loss: 135.9011 - val_mae: 9.4793
    Epoch 94/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 56.4847 - mae: 5.7130 - val_loss: 135.5290 - val_mae: 9.4594
    Epoch 95/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 51.7716 - mae: 5.5345 - val_loss: 135.1578 - val_mae: 9.4466
    Epoch 96/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 51.8409 - mae: 5.6600 - val_loss: 134.8145 - val_mae: 9.4296
    Epoch 97/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - loss: 59.7566 - mae: 5.9901 - val_loss: 134.5018 - val_mae: 9.4134
    Epoch 98/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 55.0052 - mae: 5.8298 - val_loss: 134.0967 - val_mae: 9.3984
    Epoch 99/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - loss: 52.3018 - mae: 5.8350 - val_loss: 133.6430 - val_mae: 9.3752
    Epoch 100/100
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - loss: 57.5376 - mae: 5.7228 - val_loss: 133.1667 - val_mae: 9.3560



```python
# Predict the test set results
Y_pred = model.predict(X_test)
```

    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step



```python
# Calculate MAE
mae = np.mean(np.abs(Y_test - Y_pred.flatten()))
print("Mean Absolute Error:", mae)
```

    Mean Absolute Error: 7.028558540344238



```python
# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test, y=Y_pred.flatten(), alpha=0.6)
sns.lineplot(x=Y_test, y=Y_test, color='red')
plt.xlabel("Actual MMLU Score")
plt.ylabel("Predicted MMLU Score")
plt.title("Actual vs Predicted MMLU Score with 100 epochs and 2 hidden layers")
plt.show()
```


    
![png](img/output_15_0.png)
    



```python
# Exercise: Vary the number of layers and epochs
# and observe the effect on performance

layers_to_test = [1, 2, 3]
epochs_to_test = [50, 100, 200, 500]

results = []

for num_layers in layers_to_test:
    for num_epochs in epochs_to_test:
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=input_dim))
        
        for _ in range(num_layers - 1): 
            model.add(Dense(units=32, activation='relu'))
        
        model.add(Dense(units=1, activation='linear'))  
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        history = model.fit(X_train, Y_train, validation_split=0.2, epochs=num_epochs, batch_size=32, verbose=0)
        
        Y_pred = model.predict(X_test)
        mae = np.mean(np.abs(Y_test - Y_pred.flatten()))
        
        results.append({
            'Layers': num_layers,
            'Epochs': num_epochs,
            'MAE': mae
        })

print(pd.DataFrame(results))
```

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x307530a40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m1/2[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 14ms/stepWARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x307530a40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m1/2[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 14ms/stepWARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x307530a40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step
        Layers  Epochs        MAE
    0        1      50  69.287562
    1        1     100  50.798941
    2        1     200  12.011240
    3        1     500   6.255783
    4        2      50  13.686475
    5        2     100   7.669619
    6        2     200   7.714547
    7        2     500   6.900896
    8        3      50   8.523025
    9        3     100   6.626804
    10       3     200   5.795559
    11       3     500   6.379345


Conclusion: judging from the tests, we reach the best Mean Absolute Error with 3 layers and 200 epochs.


```python
# Recreate and plot the best case scenario

# ANN with 3 hidden layers and 200 epochs
model = Sequential()
for _ in range(3): 
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=False)
Y_pred = model.predict(X_test)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test, y=Y_pred.flatten(), alpha=0.6)
sns.lineplot(x=Y_test, y=Y_test, color='red')
plt.xlabel("Actual MMLU Score")
plt.ylabel("Predicted MMLU Score")
plt.title("Actual vs Predicted MMLU Score with 200 epochs and 3 hidden layers")
plt.show()
```

    /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step



    
![png](img/output_18_2.png)
    

