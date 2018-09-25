"""
This script generate the integrity footprint on the dataset provided.
Such a footprint can be used to verify that the data has no been modified, altered or manipulated.
The integrity of the dataset can be verified in two ways: quick and deep.
The former is very fast and uses a high level type of verification such as recently modified files and file counts.
The latter basically re-compute the footprint and verifies if it matches the existing one. This is slow and should
be used only when the integrity of the dataset is a critical matter.

Structure of the dataset expected can be found at:
https://diva-dia.github.io/DeepDIVAweb/articles/prepare-dataset/
"""

# Utils
import argparse
import hashlib
import json
import logging
import os
from stat import S_ISDIR, S_ISREG
import time


def generate_integrity_footprint(dataset_folder):
    """
    This function generates the integrity footprint on the dataset provided.
    Such a footprint can be used to verify that the data has no been modified, altered or manipulated.

    The footprint file will contain the following information in a JSON format:

    {
        path : <string>                         // Path to this folder where the last step is the name of the folder
        last_modified : <date>                  // This correspond to the most recent 'last modified' in the dataset
        files : {                               // For each file
            {
                file_name : <string>            // The filename as string
                file_hash : <hash>              // This is the hash of the content
            },
            ...
        }
        folders : {                             // For each folder, recursion
           // Recursion but NO last_modified (not needed anymore)
        ]
    }

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder (see above for details)

    Returns
    -------
        A dictionary of the format explained in generate_integrity_footprint() above.
    """
    logging.info("Generating the footprint of: {}".format(dataset_folder))
    data = _process_folder(dataset_folder)
    data['last_modified'] = get_last_modified(dataset_folder)
    logging.info('Footprint generated successfully')
    return data


def get_last_modified(dataset_folder):
    """
    Elaborates the most recent 'last_modified' tag by scanning all files
    in the root folder and sub-folders.

    This routine excludes the 'footprint.json' file which, if taken into
    account, would prevent the verification process to succeed (as it modifies
    the last modified of the root itself).

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder

    Returns
    -------
        last_modified : String
            A string representing the last modified of the entire folder
    """
    # NOTE: To speed up this process it would be possible to only look
    # the last_modified of the files and folders in the root. This is
    # dangerous because if a files gets modified in the sub-folders it
    # does not modify the last_modified of its parent folder. However,
    # that would be very quick.
    last_modified = 0
    for root, folders, files in os.walk(dataset_folder):
        if 'footprint.json' in files:
            files.remove('footprint.json')
        if not files:
            continue
        tmp = max([os.path.getmtime(os.path.join(root, f)) for f in files])
        last_modified = max(tmp, last_modified)
    return str(time.ctime(last_modified))


def _process_folder(path):
    """
    Recursively descend the directory tree rooted at path,
    calling _process_file() function for each regular file

    Parameters
    ----------
    path : String (path)
        Path to folder to navigate

    Returns
    -------
        A dictionary of the format explained in generate_integrity_footprint() above.
    """
    logging.debug("Exploring folder: {}".format(path))

    # Init the dictionary to host the data
    data = {}
    data['files'] = []
    data['folders'] = []
    data['path'] = path

    # Iterate in all files into the folder
    for f in os.scandir(path=path):
        # Need to skip the footprint.json
        if f.name == 'footprint.json':
            continue
        pathname = os.path.join(path, f.name)
        mode = os.stat(pathname).st_mode
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            data['folders'].append(_process_folder(pathname))
        elif S_ISREG(mode):
            # It's a file, hash it
            data['files'].append(_process_file(pathname))
        else:
            # Unknown file type, print a message
            print('Unknown  file type, skipping %s' % pathname)
    return data


def _process_file(path):
    """
    Hashes a file and returns its filename with it in a dictionary format

    Parameters
    ----------
    path : String (path)
        Path to the file

    Returns
    -------
        A dictionary with the filename and its hash
    """
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(path, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    data = {}
    data['file_name'] = path
    data['file_hash'] = hasher.hexdigest()
    return data


def verify_integrity_quick(dataset_folder):
    """
    This function verifies that the 'last_modified' field still corresponds to the one contained in the footprint.
    This check is verify fast, but it comes at a price.
    The OS updates this number when files are added or removed to the folder, but NOT if a file is modified.
    Because of this, it is not 100% safe and especially does NOT protect you against malicious attacks!
    To have a safe check whether the data is the same you should rely on the slower
    verify_integrity_deep() function.

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder (see above for details)

    Returns
    -------
        Boolean
            Is the 'last_modified' field still actual?
    """
    logging.info("Verifying the dataset integrity - quick")
    try:
        with open(os.path.join(dataset_folder, 'footprint.json')) as json_file:
            data = json.load(json_file)
            old_timestamp = data['last_modified']
            new_timestamp = get_last_modified(dataset_folder)
            logging.info("Newly measured timestamp: {}".format(new_timestamp))
            if old_timestamp == new_timestamp:
                logging.info("Dataset integrity verified (quick). The dataset has not been modified")
                return True
            else:
                logging.error("The dataset has been modified. The last_modified field does not match: old[{}] new[{}]"
                              .format(old_timestamp, new_timestamp))
                return False
    except FileNotFoundError:
        logging.error("Missing footprint. Cannot verify dataset integrity.")
        logging.warning("Creating a new footprint, since it is missing.")
        data = generate_integrity_footprint(dataset_folder=dataset_folder)
        save_footprint(dataset_folder=dataset_folder, filename='footprint.json', data=data)
        return False


def verify_integrity_deep(dataset_folder):
    """
    This function basically re-compute the footprint and verifies if it matches the existing one.
    This is slow and should be used only when the integrity of the dataset is a critical matter.

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder (see above for details)

    Returns
    -------
        Boolean
            Is the dataset footprint still matching the data?
    """
    logging.info("Verifying the dataset integrity - deep")
    try:
        with open(os.path.join(dataset_folder, 'footprint.json')) as json_file:
            old_data = json.load(json_file)
            new_data = generate_integrity_footprint(dataset_folder)

            if old_data == new_data:
                logging.info("Dataset integrity verified (deep). The dataset has not been modified")
                return True
            else:
                logging.error("The dataset has been modified. The footprints does not match.")
                added, removed, modified, same = dict_compare(old_data, new_data)
                data = {}
                data['added'] = ', '.join(added)
                data['removed'] = ', '.join(removed)
                data['modified'] = modified
                data['same'] = ', '.join(same)
                with open(os.path.join(dataset_folder, 'differences_footprint.json'), 'w') as outfile:
                    json.dump(data, outfile)
                return False
    except FileNotFoundError:
        logging.error("Missing footprint. Cannot verify dataset integrity.")
        logging.warning("Creating a new footprint, since it is missing.")
        data = generate_integrity_footprint(dataset_folder=dataset_folder)
        save_footprint(dataset_folder=dataset_folder, filename='footprint.json', data=data)
        return False


def dict_compare(d1, d2):
    """

    Parameters
    ----------
    d1 : Dictionary
    d2 : Dictionary
        Dictionaries to compare
    Returns
    -------
        added, removed, modified, same
            Sets with the element which has been respectively added, removed, modified or stayed the same
    """
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def save_footprint(dataset_folder, filename, data):
    """
    Save the footprint on file system

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder (see above for details)
    filename : String
        Name of the file where the data will be saved
    data : dictionary
        The actual data in JSON compliant format

    Returns
    -------
        None
    """
    with open(os.path.join(dataset_folder, filename), 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        level=logging.INFO
    )

    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script generate the integrity footprint on the dataset provided')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        type=str)

    args = parser.parse_args()

    data = generate_integrity_footprint(dataset_folder=args.dataset_folder)
    save_footprint(dataset_folder=args.dataset_folder, filename='footprint.json', data=data)