"""
This file is the main entry point of DeepDIVA.

We introduce DeepDIVA: an infrastructure designed to enable quick and
intuitive setup of reproducible experiments with a large range of
useful analysis functionality. Reproducing scientific results can be
a frustrating experience, not only in document image analysis but
in machine learning in general. Using DeepDIVA a researcher can either
reproduce a given experiment or share their own experiments with others.
Moreover, the framework offers a large range of functions, such as
boilerplate code, keeping track of experiments, hyper-parameter
optimization, and visualization of data and results.

It is completely open source and accessible as Web Service through DIVAService

>> Official website: https://diva-dia.github.io/DeepDIVAweb/
>> GitHub repo: https://github.com/DIVA-DIA/DeepDIVA
>> Tutorials: https://diva-dia.github.io/DeepDIVAweb/articles.html

authors: Michele Alberti and Vinaychandran Pondenkandath (equal contribution)
"""

# Utils
import os
import subprocess
import sys
import time
import traceback
import datetime
import json
import logging
import numpy as np
from sklearn.model_selection import ParameterGrid

# SigOpt
from sigopt import Connection

# DeepDIVA
import template.CL_arguments
import template.runner
from template.setup import set_up_env, set_up_logging, copy_code
from util.misc import to_capital_camel_case, save_image_and_log_to_tensorboard
from util.visualization.mean_std_plot import plot_mean_std


########################################################################################################################
class RunMe:
    """
    This class is used as entry point of DeepDIVA.
    The there are four main scenarios for using the framework:

        - Single run: (classic) run an experiment once with the given parameters specified by
                      command line. This is typical usage scenario.

        - Multi run: this will run multiple times an experiment. It basically runs the `single run`
                     scenario multiple times and aggregates the results. This is particularly useful
                     to counter effects of randomness.

        - Optimize with SigOpt: this will start an hyper-parameter optimization search with the aid
                                of SigOpt (State-of-the-art Bayesian optimization tool). For more
                                info on how to use it see the tutorial page on:
                                https://diva-dia.github.io/DeepDIVAweb/articles.html

        - Optimize manually: this will start a grid-like hyper-parameter optimization with the
                             boundaries for the values specifies by the user in a provided file.
                             This is much less efficient than using SigOpt but on the other hand
                             is not using any commercial solutions.
    """

    # Reference to the argument parser. Useful for accessing types of arguments later e.g. setup.set_up_logging()
    parser = None

    def main(self, args=None):
        """
        Select the use case based on the command line arguments and delegate the execution
        to the most appropriate sub-routine

        Returns
        -------
        train_scores : ndarray[floats] of size (1, `epochs`) or None
            Score values for train split
        val_scores : ndarray[floats] of size (1, `epochs`+1) or None
            Score values for validation split
        test_scores : float or None
            Score value for test split
        """
        # Parse all command line arguments
        args, RunMe.parser = template.CL_arguments.parse_arguments(args)

        # Select the use case
        if args.sig_opt is not None:
            return self._run_sig_opt(args)
        elif args.hyper_param_optim is not None:
            return self._run_manual_optimization(args)
        else:
            return self._execute(args)

    def _run_sig_opt(self, args):
        """
        This function creates a new SigOpt experiment and optimizes the selected parameters.

        SigOpt is a state-of-the-art Bayesian optimization tool. For more info on how to use
        it see the tutorial page on: https://diva-dia.github.io/DeepDIVAweb/articles.html

        Parameters
        ----------
        args : dict
            Contains all command line arguments parsed.

        Returns
        -------
        None, None, None
            At the moment it is not necessary to return meaningful values from here
        """
        # Load parameters from file
        with open(args.sig_opt, 'r') as f:
            parameters = json.loads(f.read())

        # Put your SigOpt token here.
        if args.sig_opt_token is None:
            logging.error('Enter your SigOpt API token using --sig-opt-token')
            sys.exit(-1)
        else:
            conn = Connection(client_token=args.sig_opt_token)
            experiment = conn.experiments().create(
                name=args.experiment_name,
                parameters=parameters,
            )
            logging.info("Created experiment: https://sigopt.com/experiment/" + experiment.id)
            for i in range(args.sig_opt_runs):
                # Get suggestion from SigOpt
                suggestion = conn.experiments(experiment.id).suggestions().create()
                params = suggestion.assignments
                for key in params:
                    if isinstance(args.__dict__[key], bool):
                        params[key] = params[key].lower() in ['true']
                    args.__dict__[key] = params[key]
                _, _, score = self._execute(args)
                # In case of multi-run the return type will be a list (otherwise is a single float)
                if type(score) != float:
                    [conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=item)
                     for item in score]
                else:
                    conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=score)
        return None, None, None

    def _run_manual_optimization(self, args):
        """
        Start a grid-like hyper-parameter optimization with the boundaries for the
        values specifies by the user in a provided file.

        Parameters
        ----------
        args : dict
            Contains all command line arguments parsed.

        Returns
        -------
        None, None, None
            At the moment it is not necessary to return meaningful values from here
        """
        logging.info('Hyper Parameter Optimization mode')
        # Open file with the boundaries and create a grid-like list of parameters to try
        with open(args.hyper_param_optim, 'r') as f:
            hyper_param_values = json.loads(f.read())
        hyper_param_grid = ParameterGrid(hyper_param_values)
        # Run an experiment for each entry in the list of parameters
        for i, params in enumerate(hyper_param_grid):
            logging.info('{} of {} possible parameter combinations evaluated'
                         .format(i, len(hyper_param_grid)))
            for key in params:
                args.__dict__[key] = params[key]
            self._execute(args)
        return None, None, None

    @staticmethod
    def _execute(args):
        """
        Run an experiment once with the given parameters specified by command line.
        This is typical usage scenario.

        Parameters
        ----------
        args : dict
            Contains all command line arguments parsed.

        Returns
        -------
        train_scores : ndarray[floats] of size (1, `epochs`)
            Score values for train split
        val_scores : ndarray[floats] of size (1, `epochs`+1)
            Score values for validation split
        test_scores : float
            Score value for test split
        """
        # Set up logging
        # Don't use args.output_folder as that breaks when using SigOpt
        current_log_folder, writer = set_up_logging(parser=RunMe.parser, args_dict=args.__dict__, **args.__dict__)

        # Copy the code into the output folder
        copy_code(output_folder=current_log_folder)

        # Check Git status to verify all local changes have been committed
        if args.ignoregit:
            logging.warning('Git status is ignored!')
        else:
            try:
                local_changes = False
                deepdiva_directory = os.path.split(os.getcwd())[0]
                git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
                git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                git_status = str(subprocess.check_output(["git", "status"]))

                logging.debug('DeepDIVA directory is:'.format(deepdiva_directory))
                logging.info('Git origin URL is: {}'.format(str(git_url)))
                logging.info('Current branch and hash are: {}  {}'.format(str(git_branch), str(git_hash)))
                local_changes = "nothing to commit" not in git_status and \
                                "working directory clean" not in git_status
                if local_changes:
                    logging.warning('Running with an unclean working tree branch!')
            except Exception as exp:
                logging.warning('Git error: {}'.format(exp))
                local_changes = True
            finally:
                if local_changes:
                    logging.error('Errors when acquiring git status. Use --ignoregit to still run.\n'
                                  'This happens when the git folder has not been found on the file system\n'
                                  'or when the code is not the same as the last version on the repository.\n'
                                  'If you are running on a remote machine make sure to sync the .git folder as well.')
                    logging.error('Finished with errors. (Log files at {} )'.format(current_log_folder))
                    logging.shutdown()
                    sys.exit(-1)

        # Set up execution environment. Specify CUDA_VISIBLE_DEVICES and seeds
        set_up_env(**args.__dict__)

        # Select with introspection which runner class should be used.
        # Default is runner.image_classification.image_classification
        runner_class = getattr(sys.modules["template.runner." + args.runner_class],
                               args.runner_class).__dict__[to_capital_camel_case(args.runner_class)]

        try:
            # Run the actual experiment
            start_time = time.time()
            if args.multi_run is not None:
                train_scores, val_scores, test_scores = RunMe._multi_run(runner_class=runner_class,
                                                                         writer=writer,
                                                                         current_log_folder=current_log_folder,
                                                                         args=args)
            else:
                train_scores, val_scores, test_scores = runner_class.single_run(writer=writer,
                                                                                current_log_folder=current_log_folder,
                                                                                **args.__dict__)
            end_time = time.time()
            logging.info('Time taken for train/eval/test is: {}'
                         .format(datetime.timedelta(seconds=int(end_time - start_time))))
        except Exception as exp:
            if args.quiet:
                print('Unhandled error: {}'.format(repr(exp)))
            logging.error('Unhandled error: %s' % repr(exp))
            logging.error(traceback.format_exc())
            logging.error('Execution finished with errors :(')
            sys.exit(-1)
        finally:
            # Free logging resources
            logging.shutdown()
            logging.getLogger().handlers = []
            writer.close()
            print('All done! (Log files at {} )'.format(current_log_folder))
        return train_scores, val_scores, test_scores

    @staticmethod
    def _multi_run(runner_class, writer, current_log_folder, args):
        """
        Run multiple times an experiment and aggregates the results.
        This is particularly useful to counter effects of randomness.

        Here multiple runs with same parameters are executed and the results averaged.
        Additionally "variance shaded plots" gets to be generated and are visible not only
        on FS but also on tensorboard under 'IMAGES'.

        Parameters
        ----------
        runner_class : String
            This is necessary to know on which class should we run the experiments.  Default is runner.image_classification.image_classification
        writer: Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : String
            Path to the output folder. Required for saving the raw data of the plots
            generated by the multi-run routine.
        args : dict
            Contains all command line arguments parsed.

        Returns
        -------
        train_scores : ndarray[float] of size (n, `epochs`)
        val_scores : ndarray[float] of size (n, `epochs`+1)
        test_score : ndarray[float] of size (n)
            Train, Val and Test results for each run (n) and epoch
        """

        # Instantiate the scores tables which will stores the results.
        train_scores = np.zeros((args.multi_run, args.epochs))
        val_scores = np.zeros((args.multi_run, args.epochs + 1))
        test_scores = np.zeros(args.multi_run)

        # As many times as runs
        for i in range(args.multi_run):
            logging.info('Multi-Run: {} of {}'.format(i + 1, args.multi_run))
            train_scores[i, :], val_scores[i, :], test_scores[i] = runner_class.single_run(writer,
                                                                                           run=i,
                                                                                           current_log_folder=current_log_folder,
                                                                                           **args.__dict__)

            # Generate and add to tensorboard the shaded plot for train
            train_curve = plot_mean_std(arr=train_scores[:i + 1],
                                        suptitle='Multi-Run: Train',
                                        title='Runs: {}'.format(i + 1),
                                        xlabel='Epoch', ylabel='Score',
                                        ylim=[0, 100.0])
            save_image_and_log_to_tensorboard(writer, tag='train_curve', image=train_curve, global_step=i)
            logging.info('Generated mean-variance plot for train')

            # Generate and add to tensorboard the shaded plot for va
            val_curve = plot_mean_std(x=(np.arange(args.epochs + 1) - 1),
                                      arr=np.roll(val_scores[:i + 1], axis=1, shift=1),
                                      suptitle='Multi-Run: Val',
                                      title='Runs: {}'.format(i + 1),
                                      xlabel='Epoch', ylabel='Score',
                                      ylim=[0, 100.0])
            save_image_and_log_to_tensorboard(writer, tag='val_curve', image=val_curve, global_step=i)
            logging.info('Generated mean-variance plot for val')

        # Log results on disk
        np.save(os.path.join(current_log_folder, 'train_values.npy'), train_scores)
        np.save(os.path.join(current_log_folder, 'val_values.npy'), val_scores)
        logging.info('Multi-run values for test-mean:{} test-std: {}'.format(np.mean(test_scores), np.std(test_scores)))

        return train_scores, val_scores, test_scores


########################################################################################################################
if __name__ == "__main__":
    RunMe().main()
