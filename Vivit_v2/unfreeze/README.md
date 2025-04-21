This experiment has the following main features:
1. Specify the `torch.nn.MSELoss()` as loss instead of default loss of Vivit when `num_class=1`
2. Add an extra layer `torch.nn.Dropout(p=0.5)` before linear output layer in order to increase robust.
3. Use `weight_decay=0.01`
4. For the last few layers, slow down the learning rate.