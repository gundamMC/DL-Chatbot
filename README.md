# DL-Chatbot
A demo deep learning RNN chatbot using Tensorflow with a C# WPF GUI.

Note: Pretrained models are **not** provided. Please train your own network.

## Usage
### Socket
To use the python backend, run socket.py and connect to the socket on `localhost:8089`.
Then, send the user input. For example, `Hello World!`.
The socket should return the result as a simple string (no parsing needed).

### GUI 
First, start the python socket server with the directions above. Then, start the GUI and connect
to the socket with the ip `localhost` and port `8089`.
Once the socket connects, use the first tab to talk with the chatbot.

## Required python libraries:
- Tensorflow
- Numpy
- socket _(should be installed with python)_
- re _(should be installed with python)_

#### Installation
To install Tensorflow, run

```
pip install tensorflow
```

For the gpu version of Tensorflow, run

```
pip install tenorflow-gpu
```

To install Numpy, run

```
pip install numpy
```

## Training
Training a chatbot network will take a roughly a few hours. In order to train, you must provide a word embedding (the scripts by default use `glove.twitter.27B.100d.txt`) and the Cornell movie dialog corpus txts in the `./Data` folder.

To train the network, run `Train.py` and input the number of epochs to train for.
Input `save` to save the network for the socket server and `exit` to exit (_without_ saving). Input `predict` for inference.  

```
Train epochs: 10  # train 10 epochs
Train epochs: predict hey! how are you  # inference with input sequence 'hey! how are you'
Train epochs: save  # saves the network at ./model
Train epochs: 15
Train epochs: save
Train epochs: exit
```

## License
[Apache License 2.0](https://github.com/gundamMC/DL-Chatbot/blob/master/LICENSE)
