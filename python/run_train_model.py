#!/usr/bin/env python
# coding=utf-8

import argparse

from python.train_model import train_model
from python.model_util import ModelMode


def main():
    parser = argparse.ArgumentParser(description='Deep landscape forecasting training')

    parser.add_argument('--model_mode', type=ModelMode, choices=list(ModelMode), required=True, help='Model mode')
    parser.add_argument('--dataset_name', required=True, help='Dataset name')
    parser.add_argument('--dataset_path', required=False, help='Dataset path', default='../')
    args = parser.parse_args()

    train_model(
        campaign=args.dataset_name,
        model_mode=args.model_mode,
        data_path=args.dataset_path
    )


if __name__ == '__main__':
    main()
