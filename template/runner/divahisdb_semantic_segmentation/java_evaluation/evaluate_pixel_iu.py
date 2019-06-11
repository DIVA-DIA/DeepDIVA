import argparse
import itertools
import os
import re
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, STDOUT

import numpy as np


def check_extension(filename, extension_list):
    return any(filename.endswith(extension) for extension in extension_list)


def get_file_list(dir, extension_list):
    list = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if check_extension(fname, extension_list):
                path = os.path.join(root, fname)
                list.append(path)
    list.sort()
    return list


def get_score(logs):
    for line in logs:
        line = str(line)
        if "Mean IU (Jaccard index) =" in line:
            pixel_iu = line.split('=')[1][0:8]
            pixel_iu = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", pixel_iu)
            return float(pixel_iu[0])
    return None


def compute_for_all(input_img, input_gt, output_path, eval_tool):

    # Check where is the path - Debugging only
    # p = Popen(['ls'], stdout=PIPE, stderr=STDOUT)
    # logs = [line for line in p.stdout]

    # Run the JAR
    print("Starting: JAR {}".format(input_img))
    p = Popen(['java', '-jar', eval_tool,
               '-p', input_img,
               '-gt', input_gt,
               '-out', output_path,
               #'-dv'],
                ],
              stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    print("Done: JAR {}".format(input_img))
    return [get_score(logs), logs]


def get_floats(s):
    p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
    return [float(i) for i in p.findall(s)]  # Convert strings to float


def evaluate(input_folders_pred, input_folders_gt, output_path, eval_tool, j):

    # Select the number of threads
    if j == 0:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=j)

    # Get the list of all input images
    input_images = []
    for path in input_folders_pred:
        input_images.extend(get_file_list(path, ['.png','.PNG']))

    # Get the list of all input GTs
    input_gt = []
    for path in input_folders_gt:
        input_gt.extend(get_file_list(path, ['.png','.PNG']))

    # Create output path for run
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    # Debugging purposes only!
    #input_images = [input_images[9]]
    #input_gt = [input_gt[9]]
    #input_images = [input_images[0]]
    #input_gt = [input_gt[0]]
    #input_images = input_images[9:11]
    #input_gt = input_gt[9:11]

    tic = time.time()

    # For each file run
    results = list(pool.starmap(compute_for_all, zip(input_images,
                                                input_gt,
                                                itertools.repeat(output_path),
                                                itertools.repeat(eval_tool))))
    pool.close()
    print("Pool closed)")

    scores = []
    errors = []

    for item in results:
        if item[0] is not None:
            scores.append(item[0])
        else:
            errors.append(item)


    score = np.mean(scores)  if list(scores) else -1
    np.save(os.path.join(output_path, 'results.npy'), results)

    # Print the avg results
    values = []
    for item in results:
        if item[0] is not None:
            line = str(item[1][1])
            line = line.replace('NaN', '1234.4321')  # Filtered afterwards, otherwise lost information with the regexp
            values.append(np.asarray(get_floats(line)))

    values = np.vstack(values)
    values[values == 1234.4321] = float('NaN')
    print("EM={:.2f} HS={:.2f}"
          " IU={:.2f},{:.2f}[{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}]"
          " F1={:.2f},{:.2f}[{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}]"
          " P={:.2f},{:.2f}[{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}]"
          " R={:.2f},{:.2f}[{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}]"
          " Freq:[{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}]".format(*np.nanmean(values, axis=0)))

    print('Total time taken: {:.2f}, avg_pixel_iu={}, nb_errors={}'.format(time.time() - tic, score, len(errors)))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='May the odds be ever in your favor')
    # Path folders
    parser.add_argument('--input-folders-pred', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-output (e.g. /dataset/CB55/output-m /dataset/CSG18/output-m /dataset/CSG863/output-m)')

    parser.add_argument('--input-folders-gt', nargs='+', type=str,
                        required=True,
                        help='path to folders containing pixel-gt (e.g. /dataset/CB55/test-m /dataset/CSG18/test-m /dataset/CSG863/test-m)')

    parser.add_argument('--output-path', metavar='DIR',
                        required=True,
                        help='path to store output files')

    # Environment
    parser.add_argument('--eval-tool', metavar='DIR',
                        # e.g. './src/pixel_segmentation/evaluation/LayoutAnalysisEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('-j', type=int,
                        default=0,
                        help='number of thread to use for parallel search. If set to 0 #cores will be used instead')
    args = parser.parse_args()

    evaluate(**args.__dict__)
