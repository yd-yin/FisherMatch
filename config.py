import os
from utils import ensure_dirs
import glob
from os.path import join, dirname, abspath
from datetime import datetime
import configargparse


class Config(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        self.num_classes = 9

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in sorted(args.__dict__.items()):
            self.__setattr__(k, v)
            print(f"{k:20}: {v}")

        # processing
        self.cont = self.cont_ckpt is not None

        if self.is_train:
            if self.debug:
                self.exp_name, self.date = 'debug', 'debug'
            elif self.cont:
                # continue training
                self.exp_name, self.date, self.ckpt = self.cont_ckpt.split('/')
            else:
                # new training
                self.exp_name = self.get_expname()
                self.date = datetime.now().strftime('%b%d_%H%M%S')
        else:
            self.exp_name, self.date, self.ckpt = self.test_ckpt.split('/')

        print(f'exp name: {self.exp_name}')

        # log folder
        proj_root = dirname(os.path.abspath(__file__))
        print(f'proj root: {proj_root}')
        self.log_dir = join(proj_root, f'exps_{self.dataset}', self.exp_name, self.date)
        self.model_dir = join(proj_root, f'exps_{self.dataset}', self.exp_name, self.date)

        if not self.is_train or self.cont:
            assert os.path.exists(self.log_dir), f'Log dir {self.log_dir} does not exist'
            assert os.path.exists(self.model_dir), f'Model dir {self.model_dir} does not exist'
        else:
            ensure_dirs([self.log_dir, self.model_dir])

        if self.is_train:
            # save all the configurations and code
            log_name = f"log_cont_{datetime.now().strftime('%b%d_%H%M%S')}.txt" if self.cont else 'log.txt'
            py_list = sorted(glob.glob(join(dirname(abspath(__file__)), '**/*.py'), recursive=True))

            with open(join(self.log_dir, log_name), 'w') as log:
                for k, v in sorted(self.__dict__.items()):
                    log.write(f'{k:20}: {v}\n')
                log.write('\n\n')
                for py in py_list:
                    with open(py, 'r') as f_py:
                        log.write(f'\n*****{f_py.name}*****\n')
                        log.write(f_py.read())
                        log.write('================================================'
                                  '===============================================\n')

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)


    def parse(self):
        parser = configargparse.ArgumentParser(default_config_files=['settings/ssl.yml'])
        parser.add_argument('--config', is_config_file=True, help='config file path')
        self._add_basic_config_(parser)
        self._add_dataset_config_(parser)
        self._add_network_config_(parser)
        self._add_training_config_(parser)
        self._add_ssl_config_(parser)
        if not self.is_train:
            self._add_test_config_(parser)
        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        group = parser.add_argument_group('basic')
        group.add_argument('--exp_name', type=self.str2type)
        group.add_argument('--suff_name', type=str, help='name suffix appended after default exp_name')
        group.add_argument('--ss_ratio', type=float, help='supervised data ratio')
        return group

    def _add_dataset_config_(self, parser):
        group = parser.add_argument_group('dataset')
        group.add_argument('--data_dir', type=str)
        group.add_argument('--category', type=str)
        group.add_argument('--dataset', type=str, choices=['modelnet', 'pascal3d'])
        return group

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument("--network", type=str, choices=['mobilenet', 'resnet18'])
        pass
        return group

    def _add_training_config_(self, parser):
        group = parser.add_argument_group('training')
        group.add_argument('--lr', type=float, help="initial learning rate")
        group.add_argument('--batch_size', type=int, help="batch size")
        group.add_argument('--num_workers', type=int, help="number of workers for data loading")
        group.add_argument('--stage1_iteration', type=int, help='#iters of stage1')
        group.add_argument('--max_iteration', type=int, help="total number of iterations to train for supervised & merge, "
                                                             "For SSL, it is the relative number for stage2")
        group.add_argument('--log_frequency', type=int, help="visualize output every x iterations")
        group.add_argument('--val_frequency', type=int, help="run validation every x iterations")
        group.add_argument('--save_frequency', type=int, help="save models every x iterations")
        group.add_argument('--cont_ckpt', type=str, help="continue from checkpoint")
        group.add_argument('-g', '--gpu_ids', type=str)
        group.add_argument('--debug', action='store_true', help='debugging mode to avoid generating log files')
        return group

    def _add_test_config_(self, parser):
        group = parser.add_argument_group('test')
        group.add_argument('test_ckpt', type=str)
        group.add_argument('--hist_low', type=int, default=10)
        group.add_argument('--hist_high', type=int, default=150)
        return group

    def _add_ssl_config_(self, parser):
        group = parser.add_argument_group('ssl')
        group.add_argument('--SSL_lambda', type=float, help="loss = super_loss + \lambda * unsuper_loss")
        group.add_argument('--conf_thres', type=float, help="confidence threshold of the Fisher entropy")
        group.add_argument('--ulb_batch_ratio', type=float, help='the ratio of unlabeled data to labeld data in each mini-batch')
        group.add_argument('--is_ema', type=self.str2type, help='teacher parameters are EMA of student parameters or identical to student model')
        group.add_argument('--ema_decay', type=float, help='ema variable decay rate (default: 0.999)')
        group.add_argument('--type_unsuper', type=str, help='unsupervised loss', choices=['ce', 'nll'])
        return group

    def get_expname(self):
        if self.exp_name is not None:
            exp_name = self.exp_name
        else:
            name_ema = '_ema' if self.is_ema else ''
            exp_name = f'SSL{self.SSL_lambda}_{self.dataset}_{self.category}_r{self.ss_ratio}' \
                       f'_{self.type_unsuper}_thres{self.conf_thres}{name_ema}' \
                       f'_b{self.batch_size}_lr{self.lr:.1e}'
            if self.suff_name:
                exp_name += self.suff_name
        return exp_name

    @staticmethod
    def str2type(s):
        if str(s).lower() == 'true':
            return True
        elif str(s).lower() == 'false':
            return False
        elif str(s).lower() == 'none':
            return None
        else:
            return s


def get_config(phase):
    config = Config(phase)
    return config
