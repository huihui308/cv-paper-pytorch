#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import os, sys, datetime


def term_sig_handler(signum, frame):
    sys.stdout.write('\r>> {}: catched singal:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.flush()
    os._exit(0)

