import argparse
# Hyper Parameters setting
def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=3000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./model_results/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--result_name', default='./model_results/result',
                        help='Path to save matching result.')
    parser.add_argument('--model_name', default='./model_results/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='./model_results/checkpoint/t2i/...tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--feat_dim', default=32, type=int,
                        help='Dimensionality of the similarity embedding.')
    parser.add_argument('--hid_dim', default=32, type=int,
                        help='Dimensionality of the hidden state during graph convolution.')
    parser.add_argument('--out_dim', default=1, type=int,
                        help='Dimensionality of the hidden state during graph convolution.')
    parser.add_argument('--Rank_Loss', default='DynamicTopK_Negative',
                        help='DynamicTopK_Negative||Hardest_Negative||Hard_Negative')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help=' the dim of similarity embeddings.')
    parser.add_argument('--numLabel', default=1000, type=int,
                        help=' the size of category.')
    parser.add_argument('--Matching_direction', default='i2t',
                        help='i2t||t2i||ti, image-to-text matching or text-to-image matching')

    parser.add_argument('--belt', default='0.99', type=float,
                        help='percentage of weighted visual features in multi-classification ')

    parser.add_argument('--alpha', default='0.01', type=float,
                        help='percentage of semantic loss ')
    parser.add_argument('--gama', default='0.1', type=float,
                        help='percentage of classification loss ')


    opt = parser.parse_args()

    return opt
