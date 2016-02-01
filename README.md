# 3DMM_edges
This is a Matlab implementation of an algorithm for fully automatically fitting a 3D Morphable Model to a single image using landmarks and edge features.

Please note, this is in development and we are in the process of uploading the required files. Do not clone yet.

## References

If you use this code in your research, you should cite the following paper:

A. Bas, W.A.P. Smith, T. Bolkart and S. Wuhrer. "Fitting a 3D Morphable Model to Edges: A Comparison Between Hard and Soft Correspondences", to appear, 2016.

and (for the landmark detector):

X. Zhu and D. Ramanan. "Face detection, pose estimation and landmark localization in the wild" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

## Running the code



## Dependencies

In order to use this code, you need to provide your own 3D Morphable Model. One such model (and the one we used while developing the code) is the [Basel Face Model](http://faces.cs.unibas.ch/bfm/?nav=1-0&id=basel_face_model). This model is freely available upon signing a license agreement. If you use the Basel Face Model, then all you need to do is set the base path to your model in the demo file:

```matlab
BFMbasedir = '...'; % Set this to your Basel Face Model base directory
```

If you wish to use a different morphable model, this should be fine but you will need to follow these steps:

1. Your model must provide four variables:
  * shapePC is a 3n by k matrix where n is the number of model vertices and k the number of principal components
  * shapeMU is a 3n by 1 vector containing the vertices of the mean shape
  * shapeEV is a k by 1 vector containing the sorted standard deviations of each principal component (note: standard deviations not variances as the BFM name would imply)
  * tl is an f by 3 matrix containing the face list for the model
2. You need to precompute two structures that allow fast lookup of edges adjacent to vertices and faces. Two scripts are provided for doing this:

## Third party licenses

This repository ships with a copy of the [Zhu and Ramanan facial feature detector](https://www.ics.uci.edu/~xzhu/face/), which was released under the following license:

> Copyright (C) 2012 Xiangxin Zhu, Deva Ramanan
> 
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
> LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
> OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
> WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.