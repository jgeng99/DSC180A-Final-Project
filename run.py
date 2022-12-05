#!/usr/bin/env python

import argparse
import torch
import warnings

# sys.path.insert(0, 'src')

from src.models.model import *
from src.models.model_gs import *
from src.visualization.visualize import *

"""
EXAMPLE:
    parser = argparse.ArgumentParser()
    parser.add_argument("square", help="display a square of a given number",
                        type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    answer = args.square**2
    if args.verbose:
        print(f"the square of {args.square} equals {answer}")
    else:
        print(answer)
"""

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", help="log training data",
                    action="store_true")
parser.add_argument("-te", "--test", help="log test data",
                    action="store_true")
parser.add_argument("-s", "--sage", help="graphsage parameter",
                    action="store_true")
parser.add_argument("-g", "--gcn", help="gcn parameter",
                    action="store_true")
parser.add_argument("-f", "--fcn", help="fcn parameter",
                    action="store_true")
parser.add_argument("-e", "--epoch", help="epochs to run",
                    type=int)
args = parser.parse_args()

epoch = range(args.epoch)
cora_plots = []
cora_legends = []
citeseer_plots = []
citeseer_legends = []
cora_table = dict()
citeseer_table = dict()

if args.test:
    test_gcn, _ = run_gcn(path="./data/raw/test/", dataset="test",\
                          feat_suf=".features", edge_suf=".edges", epoch=epoch,\
                          task="gcn_test", to_train=args.train)
    test_fcn, _ = run_fcn(path="./data/raw/test/", dataset="test",\
                          feat_suf=".features", edge_suf=".edges", epoch=epoch,\
                          task="fcn_test", to_train=args.train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_gs = run_graphsage(2, 2, "./data/raw/", "test", "test.features",\
                                "test.edges", epoch=epoch, task="graphsage_test")

if args.sage:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cora_5 = run_graphsage(5, 128, "./data/raw/", "cora", "cora.content",\
                                   "cora.cites", epoch=epoch, task="graphsage_cora")
        citeseer_5 = run_graphsage(5, 128, "./data/raw/", "citeseer", "citeseer.features",\
                                   "citeseer.edges", epoch=epoch, task="graphsage_citeseer")

        cora_plots.append(cora_5)
        cora_legends.append("cora_graphsage_loss")
        cora_table["cora_graphsage_loss"] = cora_5[-1].item()
        citeseer_plots.append(citeseer_5)
        citeseer_legends.append("citeseer_graphsage_loss")
        citeseer_table["citeseer_graphsage_loss"] = citeseer_5[-1].item()

if args.gcn:
    gcn_cora, _ = run_gcn(path="./data/raw/cora/", dataset="cora",\
                         feat_suf=".content", edge_suf=".cites", epoch=epoch,\
                         task="gcn_cora", to_train=args.train)

    gcn_citeseer, _ = run_gcn(path="./data/raw/citeseer_int/", dataset="citeseer",\
                         feat_suf=".features", edge_suf=".edges", epoch=epoch,\
                         task="gcn_citeseer", to_train=args.train)

    cora_plots.append(gcn_cora)
    cora_legends.append("cora_gcn_loss")
    cora_table["cora_gcn_loss"] = gcn_cora[-1]
    citeseer_plots.append(gcn_citeseer)
    citeseer_legends.append("citeseer_gcn_loss")
    citeseer_table["citeseer_gcn_loss"] = gcn_citeseer[-1]

    # print(f"the final validation error for gcn is: {gcn_cora[-1]}")

if args.fcn:
    fcn_cora, _ = run_fcn(path="./data/raw/cora/", dataset="cora",\
                         feat_suf=".content", edge_suf=".cites", epoch=epoch,\
                         task="fcn_cora", to_train=args.train)
    
    fcn_citeseer, _ = run_fcn(path="./data/raw/citeseer_int/", dataset="citeseer",\
                         feat_suf=".features", edge_suf=".edges", epoch=epoch,\
                         task="fcn_citeseer", to_train=args.train)
    
    cora_plots.append(fcn_cora)
    cora_legends.append("cora_fcn_loss")
    cora_table["cora_fcn_loss"] = fcn_cora[-1]
    citeseer_plots.append(fcn_citeseer)
    citeseer_legends.append("citeseer_fcn_loss")
    citeseer_table["citeseer_fcn_loss"] = fcn_citeseer[-1]

    # print(f"the final validation error for fcn is: {fcn_cora[-1]}")

if len(cora_table) > 0:
    plot_err(cora_plots, "Cora Loss vs. Epoch", cora_legends, "epoch", "loss", "./data/out/cora_loss.png")
    plot_err(citeseer_plots, "Citeseer Loss vs. Epoch", citeseer_legends, "epoch", "loss", "./data/out/citeseer_loss.png")
    print(cora_table)
    print(citeseer_table)
