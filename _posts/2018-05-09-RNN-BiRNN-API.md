---
layout: post
title: RNN和BiRNN的API分析
categories: [深度学习, TensorFlow, Keras]
description: 分析RNN和BiRNN的API分析(针对Keras和Tensorflow)
keywords: 深度学习, TensorFlow, Keras
---

## Keras

### LSTM API

```
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

参数: 

return\_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出。

go\_backwards：布尔值，默认为False，若为True，则**逆向处理输入序列并返回逆向处理应该输出的序列（中文文档这里含糊不清，很容易给人造成误导）**。

```
源码：
LSTM调用了RNN的call函数。

RNN的call函数，存在如下代码：
last_output, outputs, states = K.rnn(step,
                                     inputs,
                                     initial_state,
                                     constants=constants,
                                     go_backwards=self.go_backwards,
                                     mask=mask,
                                     unroll=self.unroll,
                                     input_length=timesteps)

追踪到tensorflow_backend.py中的rnn函数，可以看出来输入被翻转了，输出没有看到翻转操作。
if go_backwards:
    inputs = reverse(inputs, 0)
```

[Keras LSTM go\_backwards usage](https://stackoverflow.com/questions/49946942/keras-lstm-go-backwards-usage)

[reverse the output sequence in backwards RNN](https://github.com/keras-team/keras/pull/1674)

[if go\_backward == 1, the output seqences should be reversed](https://github.com/keras-team/keras/issues/1703)

[Recurrent go\_backwards and output reversal](Recurrent go_backwards and output reversal)

### Bidirectional API

Bidirectional的backward层才是实现了的输入翻转、输出翻转。

```
Bidirectional的__init__函数。

forward和backward层go_backwards参数相反，也就是一定有一个输入翻转。
self.forward_layer = copy.copy(layer)
config = layer.get_config()
config['go_backwards'] = not config['go_backwards']
self.backward_layer = layer.__class__.from_config(config)

Bidirectional的call函数。

对forward和backward层分别得出输出。
y = self.forward_layer.call(inputs, initial_state=forward_state, **kwargs)
y_rev = self.backward_layer.call(inputs, initial_state=backward_state, **kwargs)

对y_rev的输出实现翻转。
if self.return_sequences:
    y_rev = K.reverse(y_rev, 1)
```

## Tensorflow

[RNN源码地址](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py)

### tf.nn.static\_rnn

```
static_rnn函数：

inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size, input_size]`.
returns: A pair (outputs, state) where:
    	 - outputs is a length T list of outputs (one for each input), or a nested tuple of such elements.
    	 - state is the final state
```

### tf.nn.dynamic\_rnn

```
dynamic_rnn函数：

inputs: The RNN inputs.
      默认time_major是False。
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
returns: A pair (outputs, state) where:
         outputs: The RNN output `Tensor`.
         If time_major == False (default), this will be a `Tensor` shaped:
           `[batch_size, max_time, cell.output_size]`.
         If time_major == True, this will be a `Tensor` shaped:
           `[max_time, batch_size, cell.output_size]`.
```

### tf.nn.static\_bidirectional\_rnn

```
static_bidirectional_rnn函数：

inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size], or a nested tuple of such elements.
returns: A tuple (outputs, output_state_fw, output_state_bw) where:
         outputs is a length `T` list of outputs (one for each input), which
         are depth-concatenated forward and backward outputs.
         output_state_fw is the final state of the forward rnn.
         output_state_bw is the final state of the backward rnn
```

Backward direction的输入、输出翻转。

```
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = static_rnn(
          cell_fw,
          inputs,
          initial_state_fw,
          dtype,
          sequence_length,
          scope=fw_scope)
    # Backward direction
    with vs.variable_scope("bw") as bw_scope:
      reversed_inputs = _reverse_seq(inputs, sequence_length)
      tmp, output_state_bw = static_rnn(
          cell_bw,
          reversed_inputs,
          initial_state_bw,
          dtype,
          sequence_length,
          scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)
```

### tf.nn.bidirectional\_dynamic\_rnn

[tf.nn.bidirectional\_dynamic\_rnn 官方api解释](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)

```
bidirectional_dynamic_rnn函数：

inputs: The RNN inputs.
        If time_major == False (default), this must be a tensor of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such elements.
        If time_major == True, this must be a tensor of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such elements.
returns: A tuple (outputs, output_states) where:
         outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
```

bidirectional\_dynamic\_rnn函数的outputs是**returns a tuple instead of a single concatenated Tensor**, unlike in the bidirectional\_rnn.

Backward direction的输入、输出翻转。

```
# Forward direction
output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

# Backward direction
inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)
output_bw = _reverse(
          tmp, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)

outputs = (output_fw, output_bw)
output_states = (output_state_fw, output_state_bw)
return (outputs, output_states)
```
