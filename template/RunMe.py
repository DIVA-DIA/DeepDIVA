"""
This file is the entry point of DeepDIVA.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

import json
import time
import logging
# Utils
import os
import datetime
import subprocess
import sys
import traceback

import numpy as np
# Tensor board
import tensorboardX
# SigOpt
from sigopt import Connection
# Python
from sklearn.model_selection import ParameterGrid

# DeepDIVA
import template.CL_arguments
import template.runner
from template.setup import set_up_env, set_up_logging
from util.misc import to_capital_camel_case
from util.visualization.mean_std_plot import plot_mean_std


########################################################################################################################
class RunMe:
    # TODO: improve doc
    """
    This file is the entry point of DeepDIVA.
    In particular depending on the args passed one can:
        -single run
        -multi run
        -optimize with SigOpt
        -optimize manually (grid)

    For details on parameters check CL_arguments.py
    """

    # Reference to the argument parser. Useful for accessing types of arguments later e.g. setup.set_up_logging()
    parser = None

    def main(self):
        args, RunMe.parser = template.CL_arguments.parse_arguments()

        if args.sig_opt is not None:
            self._run_sig_opt(args)
        elif args.hyper_param_optim is not None:
            self._run_manual_optimization(args)
        else:
            self._execute(args)

    def _run_sig_opt(self, args):
        # TODO: improve doc
        """
        This function creates a new SigOpt experiment and optimizes the selected parameters.

        Parameters:
        -----------
        :param args:
        :return:
            None
        """
        # Load parameters from file
        with open(args.sig_opt, 'r') as f:
            parameters = json.loads(f.read())

        # Client Token is currently Vinay's one
        conn = Connection(client_token="KXMUZNABYGKSXXRUEMELUYYRVRCRTRANKCPGDNNYDSGRHGUA")
        experiment = conn.experiments().create(
            name=args.experiment_name,
            parameters=parameters,
        )
        print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
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

    def _run_manual_optimization(self, args):
        # TODO: improve doc
        """
        Run a manual optimization search with the parameters provided


        Parameters:
        -----------
        :param args:
        :return:
            None
        """
        print('Hyper Parameter Optimization mode')
        with open(args.hyper_param_optim, 'r') as f:
            hyper_param_values = json.loads(f.read())
        hyper_param_grid = ParameterGrid(hyper_param_values)
        for i, params in enumerate(hyper_param_grid):
            print('{} of {} possible parameter combinations evaluated'.format(i, len(hyper_param_grid)))
            for key in params:
                args.__dict__[key] = params[key]
            self._execute(args)

    @staticmethod
    def _execute(args):
        # TODO: improve doc
        """

        Parameters:
        -----------
        :param args:
        :return:
        """

        # Set up logging
        # Don't use args.output_folder as that breaks when using SigOpt
        current_log_folder = set_up_logging(parser=RunMe.parser, args_dict=args.__dict__, **args.__dict__)

        # Check Git status
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
                    logging.error('Errors when acquiring git status. Use --ignoregit to still run.')
                    logging.shutdown()
                    print('Finished with errors. (Log files at {} )'.format(current_log_folder))
                    sys.exit(-1)

        # Set up execution environment
        # Specify CUDA_VISIBLE_DEVICES and seeds
        set_up_env(**args.__dict__)

        # Define Tensorboard SummaryWriter
        logging.info('Initialize Tensorboard SummaryWriter')
        writer = tensorboardX.SummaryWriter(log_dir=current_log_folder)

        # Select with introspection which runner class should be used. Default is runner.standard.Standard
        runner_class = getattr(sys.modules["template.runner." + args.runner_class],
                               args.runner_class).__dict__[to_capital_camel_case(args.runner_class)]

        try:
            start_time = time.time()
            if args.multi_run is not None:
                train_scores, val_scores, test_scores = RunMe._multi_run(runner_class, writer, current_log_folder, args)
            else:
                train_scores, val_scores, test_scores = runner_class.single_run(writer=writer, current_log_folder=current_log_folder,
                                                                                **args.__dict__)
            end_time = time.time()
            logging.info(
                'Time taken for train/eval/test is: {}'.format(datetime.timedelta(seconds=int(end_time - start_time))))
        except Exception as exp:
            if args.quiet:
                print('Unhandled error: {}'.format(repr(exp)))
            logging.error('Unhandled error: %s' % repr(exp))
            logging.error(traceback.format_exc())
            logging.error('Execution finished with errors :(')
            sys.exit(-1)
        finally:
            logging.shutdown()
            logging.getLogger().handlers = []
            writer.close()
            print('All done! (Log files at {} )'.format(current_log_folder))
            current_log_folder = None
        return train_scores, val_scores, test_scores

    @staticmethod
    def _multi_run(runner_class, writer, current_log_folder, args):
        """
        Here multiple runs with same parameters are executed and the results averaged.
        Additionally "variance shaded plots" gets to be generated and are visible not only on FS but also on
        tensorboard under 'IMAGES'.

        Parameters:
        -----------
        :param runner_class: class
            This is necessary to know on which class should we run the experiments.  Default is runner.standard.Standard

        :param writer: Tensorboard SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.

        :param args:
            Any additional arguments (especially for the runner_class)

        :return: float[n, epochs], float[n, epochs], float[n]
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
            writer.add_image('train_curve', train_curve, global_step=i)
            logging.info('Generated mean-variance plot for train')

            # Generate and add to tensorboard the shaded plot for va
            val_curve = plot_mean_std(x=(np.arange(args.epochs + 1) - 1),
                                      arr=np.roll(val_scores[:i + 1], axis=1, shift=1),
                                      suptitle='Multi-Run: Val',
                                      title='Runs: {}'.format(i + 1),
                                      xlabel='Epoch', ylabel='Score',
                                      ylim=[0, 100.0])
            writer.add_image('val_curve', val_curve, global_step=i)
            logging.info('Generated mean-variance plot for val')

        # Log results on disk
        np.save(os.path.join(current_log_folder, 'train_values.npy'), train_scores)
        np.save(os.path.join(current_log_folder, 'val_values.npy'), val_scores)
        logging.info('Multi-run values for test-mean:{} test-std: {}'.format(np.mean(test_scores), np.std(test_scores)))

        return train_scores, val_scores, test_scores


########################################################################################################################
if __name__ == "__main__":
    RunMe().main()
