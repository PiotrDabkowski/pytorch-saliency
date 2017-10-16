import sal.datasets
from sal.utils.pytorch_trainer import *
from sal.utils.pytorch_fixes import *
from sal.small import SimpleClassifier




if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sal.datasets.SUPPORTED_DATASETS.keys(), default='cifar10')
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=33)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-step', type=int, default=15)
    parser.add_argument('--save-dir', default='SmallBlackBoxModel')

    args = parser.parse_args()
    model = SimpleClassifier(base_channels=args.base)

    simple_img_classifier_train(
        model,
        sal.datasets.SUPPORTED_DATASETS[args.dataset],
        args.batch_size,
        args.epochs,
        args.lr,
        args.lr_step,
    )
    model.save(args.save_dir)