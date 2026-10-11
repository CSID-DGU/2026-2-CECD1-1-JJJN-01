"""Replay the separate folded-arm raise physics experiment."""
import argparse
from examples.m6_2 import folded_raise as f,visualize as v

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--condition',default='folded_right_turn');p.add_argument('--view',choices=['side','front'],default='side');p.add_argument('--gui',action='store_true');p.add_argument('--repeats',type=int,default=0);a=p.parse_args();v.OUT=f.OUT;v.replay(a.condition,a.view,a.gui,a.repeats)
