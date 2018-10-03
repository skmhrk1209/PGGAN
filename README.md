## Progressive Growing GAN
### TensorFlow implementation of ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/pdf/1710.10196.pdf)

### Usage
```
# Preparing datasets for training
> python data/make_dataset.py --filename train.tfrecord  --directory path/to/celeba  

# Training networks
> python main.py --train --filenames train.tfrecord  

# Generate images
> python main.py --generate --num_images 100  
```

### Generated images (128×128, cherry picked)
![0](https://user-images.githubusercontent.com/29158616/46405494-67064300-c743-11e8-8d9e-ff9fbb688828.png)
![1](https://user-images.githubusercontent.com/29158616/46405495-679ed980-c743-11e8-8b50-b3e9253953b4.png)
![2](https://user-images.githubusercontent.com/29158616/46405496-679ed980-c743-11e8-8453-4a5ebe007f55.png)
![3](https://user-images.githubusercontent.com/29158616/46405497-679ed980-c743-11e8-9785-eb33f24d7d8d.png)
![4](https://user-images.githubusercontent.com/29158616/46405499-679ed980-c743-11e8-8919-4f2e2cb5cb33.png)
![5](https://user-images.githubusercontent.com/29158616/46405500-68377000-c743-11e8-8a8a-023a42865538.png)
![6](https://user-images.githubusercontent.com/29158616/46405502-68377000-c743-11e8-923d-d11fb561183d.png)
![7](https://user-images.githubusercontent.com/29158616/46405503-68377000-c743-11e8-93c4-76729f486e13.png)
![8](https://user-images.githubusercontent.com/29158616/46405504-68377000-c743-11e8-9577-edbbe5a07e53.png)
![9](https://user-images.githubusercontent.com/29158616/46405505-68d00680-c743-11e8-9597-4f2003abaf7e.png)
![0](https://user-images.githubusercontent.com/29158616/46406215-c7967f80-c745-11e8-8845-b535043cb41b.png)
![1](https://user-images.githubusercontent.com/29158616/46406218-c7967f80-c745-11e8-84b1-450c9e0b18e8.png)
![2](https://user-images.githubusercontent.com/29158616/46406219-c82f1600-c745-11e8-9ad3-fcb6d97ac018.png)
![3](https://user-images.githubusercontent.com/29158616/46406221-c82f1600-c745-11e8-91fb-e9e3c5e48f11.png)
![4](https://user-images.githubusercontent.com/29158616/46406222-c82f1600-c745-11e8-9abb-13429d105245.png)
![5](https://user-images.githubusercontent.com/29158616/46406223-c82f1600-c745-11e8-96a8-2022b8cb5462.png)
![6](https://user-images.githubusercontent.com/29158616/46406224-c8c7ac80-c745-11e8-8208-87b2741b9047.png)
![7](https://user-images.githubusercontent.com/29158616/46406225-c8c7ac80-c745-11e8-9486-2e412209961e.png)
![8](https://user-images.githubusercontent.com/29158616/46406226-c8c7ac80-c745-11e8-92b0-cbef5a4e9e17.png)
![9](https://user-images.githubusercontent.com/29158616/46406228-c8c7ac80-c745-11e8-8088-af31327df74e.png)
![0](https://user-images.githubusercontent.com/29158616/46406587-00832400-c747-11e8-8b63-c1f294f091d0.png)
![1](https://user-images.githubusercontent.com/29158616/46406588-00832400-c747-11e8-8286-655350edeabd.png)
![2](https://user-images.githubusercontent.com/29158616/46406589-011bba80-c747-11e8-96a1-26feb04d6cb4.png)
![3](https://user-images.githubusercontent.com/29158616/46406590-011bba80-c747-11e8-96c6-0d4ee3894cf2.png)
