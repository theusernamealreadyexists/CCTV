import logging
import time
import os.path as osp
import yaml

def log_para(logger, args):
    logger.setLevel(logging.INFO)

    now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    log_name = now + '_log.txt'
    log_dir = args.snapshot_dir

    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    logger.info('--------------------------Current Settings--------------------------')
    logger.info('scripts = %s' % args.scripts)
    logger.info('lr_pretrain_img = %.7f' % args.lr_pretrain_img)
    logger.info('lr_pretrain_txt = %.7f' % args.lr_pretrain_txt)
    logger.info('lr_img = %.7f'%args.lr_img)
    logger.info('lr_txt = %.7f'%args.lr_txt)
    logger.info('cluster_number = %d'%args.cluster_number)
    logger.info('code_len = %d '%args.code_len)
    logger.info('dataset = {}'.format(args.dataset))
    logger.info('dataset_path = {}'.format(args.dataset_path))
    logger.info('num_epoch = %d' % args.num_epoch)
    logger.info('pre_num_epoch = %d' % args.pre_num_epoch)
    logger.info('eval_interval = %d' % args.eval_interval)
    logger.info('batch_size = %d' % args.batch_size)
    logger.info('momentum = %.4f' % args.momentum)
    logger.info('weight_decay = %.4f' % args.weight_decay)
    logger.info('device = %d' % args.device)
    logger.info('workers = %d' % args.workers)
    logger.info('epoch_interval = %d'% args.epoch_interval)
    logger.info('model_dir = {}'.format(args.model_dir))
    logger.info('model = {}'.format(args.model))
    logger.info('exp_name = {}'.format(args.exp_name))
    logger.info('result_dir = {}'.format(args.result_dir))
    logger.info('snapshot_dir = {}'.format(args.snapshot_dir))
    logger.info('pretrain = {}'.format(args.pretrain))
    logger.info('pretrain_dir = {}'.format(args.pretrain_dir))
    logger.info('random_train = {}'.format(args.random_train))
    logger.info('random_seed = {}'.format(args.random_seed))
    # with open(args.scripts) as f:
    #     scripts = yaml.load(f, Loader=yaml.FullLoader)
    # logger.info('LAMDBA = %.4f'%scripts['LAMDBA'])
    # logger.info('GAMMA = %.4f'%scripts['GAMMA'])
    # logger.info('KAPPA = %.4f'%scripts['KAPPA'])
    # logger.info('ALPHA_DEC = %.4f'% scripts['ALPHA_DEC'])
    # logger.info('ALPHA_HASH = %.4f'% scripts['ALPHA_HASH'])
    # logger.info('ALPHA_CADA = %.4f'% scripts['ALPHA_CADA'])
    # logger.info('--------------------------------------------------------------------')

# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler(".log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")