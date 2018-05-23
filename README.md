# AI-Chatbot
A demo deep learning chatbot using Tensorflow with a C# WPF GUI.

## Usage
### Socket
To use the python backend, run socket.py and connect to the socket on `localhost:8089`.
Then, send the user input with `input: ` at the beginning. For example, `input: Hello World!`.
The socket should return just the result (no parsing needed).

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
It is recommended to use the pre-trained network provided in the backend folder.
Training a new network from scratch will require a lot of memory and computing power. **Around 5GB VRAM and 2-3GB RAM**
(The network is optimized for Tensorflow-GPU and runs on both GPU and CPU. To change this, modify `network.py`)

To train the network, run `Train.py` and input the number of epochs to train for.
Input `save` to save the network for the socket server and `exit` to exit (_without_ saving).
For example, the following inputs will save the network after 50 epochs and another 100 epochs

```
Train epochs: 50
Train epochs: save
Train epochs: 100
Train epochs: save
Train epochs: exit
```

## License
[Apache License 2.0](https://github.com/gundamMC/AI-Chatbot/blob/master/LICENSE)