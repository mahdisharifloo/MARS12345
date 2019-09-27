# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../utils/')
#import utils and toolbox
import global_feature_extraction

#---------
#set global values
#---------
train_path = 'dataset/train'
test_path  = 'dataset/test'
fixed_size = tuple((500, 500))

