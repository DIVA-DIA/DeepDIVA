# DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments

We introduce DeepDIVA: an infrastructure designed to enable quick and
intuitive setup of reproducible experiments with a large range of useful
analysis functionality.
Reproducing scientific results can be a frustrating experience, not only
in document image analysis but in machine learning in general.
Using DeepDIVA a researcher can either reproduce a given experiment with
a very limited amount of information or share their own experiments with
others.
Moreover, the framework offers a large range of functions, such as
boilerplate code, keeping track of experiments, hyper-parameter
optimization, and visualization of data and results.
To demonstrate the effectiveness of this framework, this paper presents
case studies in the area of handwritten document analysis where
researchers benefit from the integrated functionality.
DeepDIVA is implemented in Python and uses the deep learning framework
PyTorch.
It is completely open source and accessible as Web Service through
[DIVAServices](http://divaservices.unifr.ch).

## Additional resources

- [Fancy page](https://github.com/DIVA-DIA/DeepDIVA)
- [Tutorials](https://github.com/DIVA-DIA/DeepDIVA)
- [Paper on arXiv](https://github.com/DIVA-DIA/DeepDIVA)



## Getting started

### Example of problem hunting

Below there is an example supporting the usefulness of overlapping the prediction quality visualization with the original image.
Focus on the red pixels pointed at by the white arrow: they are background pixels mis-classified as foreground.
In the normal visualization (left) its not possible to know why would an algorithm decide that in that
spot there is something belonging to foreground, as it is clearly far from regular text.
However, when overlapped with the original image (right) one can clearly see that in this area there is an
ink stain which could explain why the classification algorithm is deceived into thinking these pixel were
foreground. This kind of interpretation is obviously not possible without the information provided by the
original image like in (right).

![Alt text](examples/visualization_error.png?raw=true)
![Alt text](examples/overlap_error.png?raw=true)

- GREEN:   Foreground predicted correctly
- YELLOW:  Foreground predicted - but the wrong class (e.g. Text instead of Comment)
- BLACK:   Background predicted correctly
- RED:     Background mis-predicted as Foreground
- BLUE:    Foreground mis-predicted as Background

```
 -gt,--groundTruth <arg>      Ground Truth image
 -p,--prediction <arg>        Prediction image
 -o,--original <arg>          (Optional) Original image, to be overlapped with the results visualization
 -j,--json <arg>              (Optional) Json Path, for the DIVAServices JSON output
 -out,--outputPath <arg>      (Optional) Output path (relative to prediction input path)
 -dv,--DisableVisualization   (Optional)(Flag) Vsualizing the evaluation as image is NOT desired
 ```
**Note:** this also outputs a human-friendly visualization of the results next to the
 `prediction_image.png` which can be overlapped to the original image if provided
 with the parameter `-overlap` to enable deeper analysis.

