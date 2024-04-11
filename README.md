# Greedy Grid Search: A 3D Registration Baseline

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/challenging-the-universal-representation-of/point-cloud-registration-on-fpv1)](https://paperswithcode.com/sota/point-cloud-registration-on-fpv1?p=challenging-the-universal-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/challenging-the-universal-representation-of/point-cloud-registration-on-kitti-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti-trained-on?p=challenging-the-universal-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/challenging-the-universal-representation-of/point-cloud-registration-on-eth-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-eth-trained-on?p=challenging-the-universal-representation-of)

<br>

⚠️⚠️⚠️ An improved version of the basline has been introduced [here](https://github.com/DavidBoja/exhaustive-grid-search). Please use the new baseline benchmark of this one. ⚠️⚠️⚠️

<br>

This Github presents the code for the following paper: [Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration](https://arxiv.org/abs/2211.16301) presented at the BMVC 2022 workshop (URCV 22).

<p align="center">
  <img src="https://github.com/DavidBoja/greedy-grid-search/blob/main/assets/pipeline-image.png" width="950">
</p>

## TL;DR

We analyze the problem of 3D registration and highlight 2 main issues:
1. Learning-based methods struggle to generalize onto unseen data
2. The current 3D registration benchmark datasets suffer from data variability

We address these problems by:
1. Creating a simple baseline model that outperforms most state-of-the-art learning-based methods
3. Creating a novel 3D registration benchmark FPv1 (called FAUST-partial in the paper) based on the FAUST dataset

## Data
#### 3DMatch
Download the testing examples from [here](https://3dmatch.cs.princeton.edu/) under the title `Geometric Registration Benchmark` --> `Downloads`. There are 8 scenes that are used for testing. In total, there are 16 folders, two for each scene with names `{folder_name}` and `{folder_name}-evaluation`:
```
7-scenes-redkitchen
7-scenes-redkitchen-evaluation
sun3d-home_at-home_at_scan1_2013_jan_1
sun3d-home_at-home_at_scan1_2013_jan_1-evaluation
sun3d-home_md-home_md_scan9_2012_sep_30
sun3d-home_md-home_md_scan9_2012_sep_30-evaluation
sun3d-hotel_uc-scan3
sun3d-hotel_uc-scan3-evaluation
sun3d-hotel_umd-maryland_hotel1
sun3d-hotel_umd-maryland_hotel1-evaluation
sun3d-hotel_umd-maryland_hotel3
sun3d-hotel_umd-maryland_hotel3-evaluation
sun3d-mit_76_studyroom-76-1studyroom2
sun3d-mit_76_studyroom-76-1studyroom2-evaluation
sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation
```
We use the overlaps from PREDATOR [1] (found in `data/overlaps`) to filter the data and use only those with overlap > 30%.

#### KITTI
Download the testing data from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) under `Download odometry data set (velodyne laser data, 80 GB)`. 3 scenes are used for testing:
```
08
09
10
```
Download the `test.pkl` from GeoTransformer [3] [here](https://github.com/qinzheng93/GeoTransformer/blob/main/data/Kitti/metadata/test.pkl) and put it in the same directory where the scenes are located.

#### ETH
Download the testing data from [here](https://github.com/zgojcic/3DSmoothNet#eth). There are 4 scenes that are used for testing:
```
gazeebo_summer
gazeebo_winter
wood_autumn
wood_summer
```
We use the overlaps from Perfect Match [2] (found in `data/overlaps`) to filter the data and use only those with overlap > 30%. We obtain the overlaps from their `overlapMatrix.csv` in each scene.

#### FPv1
Download the FAUST scans from [here](http://faust.is.tue.mpg.de/challenge/Inter-subject_challenge/datasets). There are 100 scans in the training dataset named `tr_scan_xxx.ply` that are used for the registration benchmark. To use the same benchmark as in the paper, download the folder `FPv1` from [here](https://ferhr-my.sharepoint.com/:f:/g/personal/dbojanic_fer_hr/EgH5iaoUDp1PmL1K8xBDnCQBXU82ZlrSG_PiZmlIEK7dwQ?e=2U9EtJ).
To create your own benchmark, we provide a toolbox [github.com/DavidBoja/FPv1](https://github.com/DavidBoja/FPv1)

## Running baseline
We provide a Dockerfile to facilitate running the code. Run in terminal:

```bash
cd docker
sh docker_build.sh
sh docker_run.sh CODE_PATH DATA_PATH
```
by adjusting the CODE_PATH and DATA_PATH. These are paths to volumes that are attached to the container. The CODE_PATH is the path to the clone of this github repository, while the DATA_PATH is the location of all the data from data section in this documentation. 

You can attach to the container using
```bash
docker exec -it ggs-container /bin/bash
```

Next, change the `DATASET-PATH` for each dataset in `config.yaml`.

Next, once inside the container, you can run:
```python
python register.py -D xxx
```

where xxx can be 3DMatch, KITTI, ETH or FP (indicating FPv1). The script saves the registration results in results/timestamp, where timestamp changes according to the time of script execution.


## Running refinement
To refine the results from the baseline registration, we provide a script that runs one of the three ICP algorithms:
- p2point icp
- p2plane icp
- generalized icp

To choose between the three algorithms and set their parameters, adjust the `REFINE` option in `config.yaml`

```python
python refine.py -R results/timestamp
```
where timestamp should be changed to the baseline results path you want to refine.


## Evaluate
Similarly to the refinement above, to evaluate the registration you can run:
```python
python evaluate.py -R results/timestamp
```
where timestamp should be changed accordingly to indicate your results.

## Running demo
We additionally provide a demo script `demo.py` for registering two arbitrary point clouds. First, adjust the parameters in `config.yaml` under the `DEMO` category. Next, you can run 
```python
python demo.py  --pc_target_pth pc_target.ply --pc_source_pth pc_source.ply
```
where you need to specify the target and source point cloud paths.


## Citation

If you use our work, please reference our paper:

```
@inproceedings{Bojanić-BMVC22-workshop,
   title = {Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration},
   author = {Bojani\'{c}, David and Bartol, Kristijan and Forest, Josep and Gumhold, Stefan and Petkovi\'{c}, Tomislav and Pribani\'{c}, Tomislav},
   booktitle={BMVC 2022 Workshop Universal Representations for Computer Vision},
   year = {2022}
   url={https://openreview.net/forum?id=tJ5jWBIAbT}
}
```

# ToDo
- [X] Update documentation
- [X] Update data documentation
- [ ] Add results section to documentation
- [X] Demo script

## References 
[1] [PREDATOR](https://github.com/prs-eth/OverlapPredator)
[2] [Perfect Match](https://github.com/zgojcic/3DSmoothNet)
[3] [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
