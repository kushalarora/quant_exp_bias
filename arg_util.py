import os
from argparse import ArgumentParser

def get_args():
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Number of ngrams to consider for Reward.')
    parser.add_argument('--embedding_size', type=int, default=200,
                        help='Number of ngrams to consider for Reward.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='initial learning rate')
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--episodes', type=int, default=8000000,
                        help='Number of episodes.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tie_weights', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--context_size', type=int, default=4,
                        help='Number of ngrams to consider for Reward.')
    parser.add_argument('--nonlinearity', type=str, default='relu',
                        help='Non linearity for rnn.')
    parser.add_argument('--max_len', type=int, default=40,
                        help='Maximum number of words in a sentence.')
    parser.add_argument('--validate_freq', type=int, default=1000,
                        help='Validate Frequency.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Log to tensorboard')
    parser.add_argument('--log_dir', type=str, default='./log/',
                        help='Log directory for tensorboard')
    parser.add_argument('--print_sentence', action='store_true',
                        help='Print Sentence.')
    parser.add_argument('--lm_path', type=str, default='data/penn/train.arpa',
                        help='LM path.')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='GPU ID.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch of trajectories before optimizing')
    parser.add_argument('--config', '-c', type=str, default=None, 
                        help='JSON file with argument for the run.')
    parser.add_argument('--debug', '-d', action='store_true', 
                        help='Are we in debug mode right now.')
    args = parser.parse_args()
    return args

