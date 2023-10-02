import tensorflow as tf
import os
import train
import preprocess

import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description='TensorFlow 2.x Args Parser')
    
    parser.add_argument('--train_json', type=str, default="train_record", help='the processed train json file')
    parser.add_argument('--test_json', type=str, default="test_record", help='the processed test json file')
    parser.add_argument('--train_meta', type=str, default="train_meta", help='the processed train number')
    parser.add_argument('--test_meta', type=str, default="test_meta", help='the processed test number')
    parser.add_argument('--log_dir', type=str, default="log_dir", help='where to save the log')
    parser.add_argument('--model_dir', type=str, default="log_dir", help='where to save the model')
    parser.add_argument('--data_dir', type=str, default="data_dir", help='where to read data')
    parser.add_argument('--class_num', type=int, default=18, help='the class number')
    parser.add_argument('--length_block', type=int, default=1, help='the length of a block')
    parser.add_argument('--min_length', type=int, default=2, help='the flow under this parameter will be filtered')
    parser.add_argument('--max_packet_length', type=int, default=5000, help='the largest packet length')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='ratio of train set of target app')
    parser.add_argument('--keep_ratio', type=float, default=1, help='ratio of keeping the example (for small dataset test)')
    parser.add_argument('--max_flow_length_train', type=int, default=200, help='the max flow length, if larger, drop')
    parser.add_argument('--max_flow_length_test', type=int, default=2000, help='the max flow length, if larger, drop')
    parser.add_argument('--test_model_dir', type=str, default="log_dir", help='the model dir for test result')
    parser.add_argument('--pred_dir', type=str, default="pred_dir", help='the dir to save predict result')
    parser.add_argument('--batch_size', type=int, default=128, help='train batch size')
    parser.add_argument('--hidden', type=int, default=128, help='GRU dimension of hidden state')
    parser.add_argument('--layer', type=int, default=2, help='layer number of length RNN')
    parser.add_argument('--length_dim', type=int, default=16, help='dimension of length embedding')
    parser.add_argument('--length_num', type=str, default='auto', help='length_num')
    parser.add_argument('--keep_prob', type=float, default=0.8, help='the keep probability for dropout')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--iter_num', type=int, default=int(1.2e5), help='iteration number')
    parser.add_argument('--eval_batch', type=int, default=77, help='evaluated train batches')
    parser.add_argument('--train_eval_batch', type=int, default=77, help='evaluated train batches')
    parser.add_argument('--decay_step', type=str, default='auto', help='the decay step')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='the decay rate')
    parser.add_argument('--mode', type=str, default='train', help='model mode: train/prepro/test')
    parser.add_argument('--capacity', type=int, default=int(1e3), help='size of dataset shuffle')
    parser.add_argument('--loss_save', type=int, default=100, help='step of saving loss')
    parser.add_argument('--checkpoint', type=int, default=5000, help='checkpoint to save and evaluate the model')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Global Norm gradient clipping rate')
    parser.add_argument('--is_cudnn', type=bool, default=True, help='whether take the cudnn gru')
    parser.add_argument('--rec_loss', type=float, default=0.5, help='the parameter to control the reconstruction of length sequence')
    
    args = parser.parse_args()
    return args


def main(_):
    config = get_args()
    if config.length_num == 'auto':
        config.length_num = config.max_packet_length // config.length_block + 4
    else:
        config.length_num = int(config.length_num)
    if config.decay_step != 'auto':
        config.decay_step = int(config.decay_step)
    if config.mode == 'train':
        train.train(config)
    elif config.mode == 'prepro':
        preprocess.preprocess(config)
    elif config.mode == 'test':
        print(config.test_model_dir)
        train.predict(config)
    else:
        print('unknown mode, only support train now')
        raise Exception


if __name__ == '__main__':
    tf.app.run()
