import glob
import os
from semiparametrictransfer.utils.construct_html import fill_template
from semiparametrictransfer.utils.construct_html import save_html_direct
import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_dirs(pattern):
    names = sorted_nicely(glob.glob(pattern))
    # remove /parent from every path
    # names = ['/'+ '/'.join(str.split(n, '/')[2:]) for n in names]
    return names

def make_exp_table(startgoal_dir, target_dir):
    goal_im_filenames = get_dirs(startgoal_dir)
    if len(goal_im_filenames) == 0:
        raise ValueError('no startgoal pairs found!')

    item_dict = {}

    num_gifs = len(glob.glob(target_dir + '/traj*/video.gif'))
    if num_gifs == 0:
        raise ValueError('no gifs found!')

    for i, goal_im in zip(range(num_gifs), goal_im_filenames):
        trajname = 'traj{}'.format(i)
        os.system('cp {} {}'.format(goal_im, os.path.join(target_dir, trajname)))
        item_dict['trial{}'.format(i)] = [trajname + '/video.gif', trajname + '/im_30.png']

    print('found {} trajectories'.format(i))

    html = fill_template(item_dict, title_fow_func=_format_title_row, exp_name='BC from States')
    save_html_direct(os.path.join(target_dir, 'vis_exp.html'), html)


def _format_title_row(title, length):
    template = "  <tr>\n    <th></th>\n"
    cols = ['gif', 'goal']
    for colname in cols:
        template += "    <th> {} </th>\n".format(colname)
    template += "  </tr>\n"
    return template


if __name__ == '__main__':
    startgoal_dir = os.path.join(os.environ['DATA'] + '/spt_trainingdata', 'sim/tabletop-texture-benchmark/raw/traj_group*/traj*/images0/im_30.png')
    target_dir = os.path.join(os.environ['EXP'] + '/spt_experiments', 'spt_control_experiments/control/trajfollowing/verbose')
    make_exp_table(startgoal_dir, target_dir)
