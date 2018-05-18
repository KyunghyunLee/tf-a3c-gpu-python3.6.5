# tf-a3c-gpu(forked from https://github.com/hiwonjoon/tf-a3c-gpu)

Tensorflow implementation of A3C algorithm using GPU (haven't tested, but it would be also trainable with CPU).

On the original paper, ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783),
suggests CPU only implementations, since environment can only be executed on CPU which causes unevitable communication
overhead between CPU and GPU otherwise.

I think his work(t3-a3-gpu) is nice because of saving my time more three times. Thank you!

## Hyperparamter
| Hyperparameter    |                                             |
|-------------------|---------------------------------------------|
| Optimization      | ADAM                                        |
| Learning rate     | 1e-4                                        |
| Gradient Clipping | Global gradient clipping: 1.0               |
| Reward            | Average Reward: 418.8 Maximum Reward: 851.0 |

## Requirements

- Python 2.7
- Tensorflow v1.2
- OpenAI Gym v0.9
- scipy, pip (for image resize)
- tqdm(optional)
- better-exceptions(optional)
- opencv-python (pip)

## Training from scratch

- All the hyperparmeters are defined on `a3c.py` file. Change some hyperparameters as you want, then execute it.
```
python ac3.py
```

## Validation with trained models

- If you want to see the trained agent playing, use the command:
```
python ac3-test.py --model ./models/breakout-v0/last.ckpt --out /tmp/result
```


## Notes & Acknowledgement

- Here is other implementations and code I refer to.
    - [ppwwyyxx's implementation](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym)
    - [carpedm20's implementation of DQN](https://github.com/carpedm20/deep-rl-tensorflow)
