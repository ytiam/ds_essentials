
'''
	Check all the python module installed successfully.


'''


import sys
import traceback


try:

    #NLP
    from flair.data import Sentence
    from flair.models import SequenceTagger

    # # NLU
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN


    from rasa_nlu.training_data import load_data
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.model import Trainer
    from rasa_nlu import config

    from allennlp.data import Instance
    from allennlp.data.fields import TextField, SequenceLabelField
    from allennlp.data.dataset_readers import DatasetReader
    from allennlp.common.file_utils import cached_path
    from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
    from allennlp.data.tokenizers import Token
    from allennlp.data.vocabulary import Vocabulary
    from allennlp.models import Model
    from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
    from allennlp.modules.token_embedders import Embedding
    from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
    from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
    from allennlp.training.metrics import CategoricalAccuracy
    from allennlp.data.iterators import BucketIterator
    from allennlp.training.trainer import Trainer


    import textacy
    import textacy.keyterms

    import nlpnet



    from termcolor import colored
    
    import gc

    from multiprocessing import Pool
    import bisect

    from configparser import ConfigParser


    import sys
    from os import path, listdir, makedirs, getcwd, chdir
    import pandas as pd
    import time, re
    from subprocess import check_output, call
    from datetime import datetime
    import ast
    import multiprocessing as mp
    import multiprocessing
    import concurrent.futures
    import shutil
    from tqdm import tqdm
    
    from ast import literal_eval
    from collections import Counter
    import pdb
    import urllib.parse
    import collections


    from setuptools import setup, find_packages
    from bs4 import BeautifulSoup   

    import requests


    from flashtext import KeywordProcessor

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process



    import psycopg2
    import logging
    from sqlalchemy import create_engine

    #
    import lightgbm as lgb



  

except Exception as error:
        e, value, tb  = sys.exc_info()
        print(traceback.print_exception(e ,value, tb))
