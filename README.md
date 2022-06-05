# pointnet-tf2.0-customize

pointnet-tf2.0-customize

## Custom data training situation visualization

![custom](https://github.com/iubizi/pointnet-tf2.0-customize/blob/main/visualization%40ModelNet40.png)

# File meaning and purpose

- `2048_2-8` Data obtained from matlab, where 1-4 are positive and 5-8 are negative. It contains a py file that can be used to merge mat data.
- `customize.npz` Compressed training data
- `pointnet.py` training program
- `visualization@ModelNet40.png` Loss and accuracy in training (each epoch)
- `pointnet_weights@customize.h5` Saved weight, there is no model because the model cannot be serialized.
