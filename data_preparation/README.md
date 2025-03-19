# Prepare Your Training Data

To prepare your training data, you have to prepare two stuffs.

- A scenes.json containing all the scene_id of your data.
- A directory storing all your scenes.

The example JSON file can be found at data/lvis.json.

The datasets should be organized as follows:

```
Your_dataset---------------
    |scene_0001
    |scene_0002
    |...
    |scene_9999
```

And your json file should be organzied as:
```
[
    "scene_0001",
    "scene_0002",
    ...
    "scene_9999"
]
```

Please refer to data_example/scene_0001 for an example of the data you should prepare.
In the meta.json of each scene, you need to prepare:
- camera_angle_x: FOV.
- transform_matrix: the c2w matrix for each view.
- The path to the rgb and depth.

Remember to change the data path in lrm/datasets.py!