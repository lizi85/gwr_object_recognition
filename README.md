# GWR Object Recognition

Creator: Luiza Mici 

.. this work is part of my PhD

### Dependecy installation

The code is tested with python 2.7

1) install [miniconda](https://conda.io/miniconda.html)
2) install python requirements ``` pip install -r requirements.txt ```
3) install cyvlfeat ``` conda install -c menpo cyvlfeat ```
4) istall imageio ``` conda install -c conda-forge imageio ```
5) (optional) ``` conda install -c conda-forge matplotlib ```

### Testing with mnist
1) install ``` pip install mnist  ```


#### Object recognition

Create your own implementation of 
```python
 from dataset import Dataset
 
 class MyDataset(Dataset):
    def __init__(self):
        # TODO: my config based on Dataset filds
 
 ```

### References

- Marsland, S., Shapiro, J., and Nehmzow, U. (2002). A self-organising network that grows when required. Neural Networks, 15(8-9):1041-1058.
- Delhumeau, J., Gosselin, P.H., Jegou, H. and Perez, P., 2013, October. Revisiting the VLAD image representation. In Proceedings of the 21st ACM international conference on Multimedia (pp. 653-656). ACM.
- Mici, L., Parisi, G. I., Wermter, S. (2018) A self-organizing neural network architecture for learning human-object interactions. Neurocomputing, 307:14--24. 