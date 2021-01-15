# FPGA-based-LSTM
A novel FPGA-based intent recognition system utilizing deep recurrent neural networks.

This repository contains the open-source C++ code of Vivado HLS/Vitis and the appropriate dataset for the FPGA acceleration of a RNN-LSTM model presented in the following paper:

"[omitted due to ongoing blind review; available on request]".

The folder "dataset" contains all the weights and biases in .csv files of the trained model that will be used as inputs to the FPGA implementation. Moreover, the testing data and the results produced by the original Python code duting training are included.

The "rnn_model_tb.cpp", "rnn_model.cpp" and "rnn_LSTM.cpp" contain the three source code of the FPGA implementation in Vivado HLS/Vitis. 
The first one presents the testbench which imports the .csv files into simple tables and calls for the execution of the optimized FPGA-based model.
The last two are the complete optimazed FPGA-based model which achieves excellent accelaration and energy efficiency. Specifically, the "rnn_LSTM.cpp" contains the optimized LSTM layers of the model and the "rnn_model.cpp" the rest of the layers.

In order to reproduce the same results as the corresponding paper, the following steps must be performed:
1. For each .cpp file create a seperate file in Vivado HLS/Vitis and copy its contends in the Vivado HLS/Vitis files. The "rnn_model_tb.cpp" contains the testbench therefore must be included in the appropriate testbench of the project.
2. Download and save in a accessable place the "dataset" folder. Afterwards, in the testbench file edit the present address of the folder in all previous addresses. Therefore, the model can import the correct inputs and produce the propriate results.
3. The model is ready to be tested in Vivado HLS/Vitis.




