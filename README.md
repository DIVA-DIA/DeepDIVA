# LayoutAnalysisEvaluator
Layout Analysis Evaluator for the ICDAR 2017 competition on Layout Analysis for Challenging Medieval Manuscripts

Minimal usage: `java -jar LayoutAnalysisEvaluator.jar -gt gt_image.png -p prediction_image.png`

Parameters list: utility-name
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

## Visualization of the results

Along with the numerical results (such as the Intersection over Union (IU), precision, recall,F1)
the tool provides a human friendly visualization of the results.
Additionally, when desired one can provide the original image and it will be overlapped with
the visualization of the results.
This is particularly helpful to understand why certain artifacts are created.
The three images below represent the three steps: the original image, the visualization of the result
and the two overlapped.

![Alt text](images/confusionMatrix.png?raw=true)
![Alt text](images/histograms.png?raw=true)
![Alt text](images/shadyPlot.png?raw=true)

### Interpreting the colors

Pixel colors are assigned as follows:

- GREEN:   Foreground predicted correctly
- YELLOW:  Foreground predicted - but the wrong class (e.g. Text instead of Comment)
- BLACK:   Background predicted correctly
- RED:     Background mis-predicted as Foreground
- BLUE:    Foreground mis-predicted as Background

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
