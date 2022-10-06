# Parallelizing the convolution operation using MPI

Project work for "Sistemi concorrenti" course at Unisannio. This repository was forked from [0xnirmal/Parallel-Convolution-MPI](https://github.com/0xnirmal/Parallel-Convolution-MPI) and the documentation about the convolution operation has been moved in [docs](https://github.com/luigiocone/Parallel-Convolution-MPI/tree/master/docs) folder.

The goal of this project is to optimize the convolution operation in order to minimize the execution time. All changes must be made taking into account the machine's specs on which the program will run. Some of those specs will be reported in the next section.

## Computational nodes - Basic info
There are 4 different nodes in the execution environment. Those nodes have the following configuration:
| Node spec | Value |
|--|--|
| CPU | (2x) Intel Xeon E5-2650V2 (3.40 GHz) |
| Hyper-Threading | On some node is turned on |
| RAM | 64 GB |
| Disk | 1 TB |

Each node is built with 2 Xeon processors (_Intel Xeon E5-2650V2_) meaning 16 physical core for each node.
| Xeon spec | Quantity |
|--|--|
| Cores/Threads | 8/16 |
| L1 Cache | 64KB per core (32 KB 8-way set associative instruction caches + 32 KB 8-way set associative data caches) |
| L2 Cache | 256 KB per core (256 KB 8-way set associative caches) |
| L3 Cache | 20 MB shared cache |

Also, nodes are interconnected with a low-latency network (InfiniBand) at 56 Gb/sec bandwidth and a Fast Ethernet at 100 Mb/s. The OS is _Rocks 7.0 - Manzanita_.

## How to run
Dependencies: 
- `MPI` (e.g. MPICH) 
- `PAPI` 

Once installed, digit: 
```
make && ./run.sh
```
If source matrix is an image, source and result images could be plot with python:
```
cd io-files
python3 image.py
```
The following image is an output example. Left image is the source matrix, the right image has been computed (after one convolution iteration) using a _ridge detection kernel_.
![Ridge detection (normalized)](https://github.com/luigiocone/Parallel-Convolution-MPI/blob/master/docs/img/ridge_detection_camera_normalization.png?raw=true)

The previous image was computed using also a normalization process (as described [here](https://it.mathworks.com/help/vision/ref/2dconvolution.html)), without the normalization the result would be the following: 
![Ridge detection](https://github.com/luigiocone/Parallel-Convolution-MPI/blob/master/docs/img/ridge_detection_camera.png?raw=true)
