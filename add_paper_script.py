# Added a new paper to the list.
# Author: Yinyu Nie
# Updated data: 23 June, 2021

# Usage:
# see python add_paper_script.py -h
# python -t 3 -p CVPR2020 -l https://arxiv.org/abs/2002.12212 -ti "Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image" -o pytorch=https://github.com/yinyunie/Total3DUnderstanding Project=https://yinyunie.github.io/Total3D/
# if the link argument [-l] is given an arxiv link, you do not need to give the title argument [-ti], like
# python -t 3 -p CVPR2020 -l https://arxiv.org/abs/2002.12212 -o pytorch=https://github.com/yinyunie/Total3DUnderstanding Project=https://yinyunie.github.io/Total3D/
# [-o] is an argument with variable length. You can give a dictionary like above.

import sys
import argparse
from libs import ParseKwargs, existing_topics
import urllib.request
import re

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='A script for adding new paper. For example, python -t 3 -p CVPR2020 -l https://arxiv.org/abs/2002.12212 -ti "Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image" -o pytorch=https://github.com/yinyunie/Total3DUnderstanding Project=https://yinyunie.github.io/Total3D/', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--list', type=str, default='./README.md',
                        help='Paper list file.')
    parser.add_argument('--topic', '-t', type=int,
                        help = '''
                        [0] 3D Detection & Segmentation;
                        [1] Shape Representation;
                        [2] Shape & Scene Completion;
                        [3] Shape Reconstruction;
                        [4] 3D Scene Understanding;
                        [5] 3D Scene Reconstruction;
                        [6] NeRF;
                        [7] About Human Body;
                        [8] General Methods;
                        [9] Others (inc. Networks in Classification, Matching, Registration, Alignment, Depth, Normal, Pose, Keypoints, etc.);
                        [10] Survey, Resources and Tools.
                        ''')
    parser.add_argument('--pub', '-p', type=str, help='Publication info (e.g. Arxiv, CVPR2021).')
    parser.add_argument('--link', '-l', type=str, help='Paper link (e.g. https://arxiv.org/pdf/2002.12212.pdf).')
    parser.add_argument('--title', '-ti', type=str, help='Paper tile.')
    parser.add_argument('--others', '-o', nargs='*', type=str, help='other useful links', action=ParseKwargs)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    topic_name = existing_topics[args.topic]
    pub_name = args.pub
    link_addr = args.link

    if 'arxiv.org' in link_addr:
        if link_addr.endswith('.pdf'):
            arxiv_id = '.'.join(link_addr.split('/')[-1].split('.')[:-1])
            link_addr = 'https://arxiv.org/abs/' + str(arxiv_id)
        else:
            arxiv_id = link_addr.split('/')[-1]
        html_text = urllib.request.urlopen(link_addr).read()
        html_text = html_text.decode("utf8")
        pattern = re.compile(r'Title:.*?</span>(.+?)</h1>')
        re_result = re.search(pattern, html_text)
        paper_title = re_result.group(1)
        pub_date = int(arxiv_id.split('.')[0])
    else:
        paper_title = args.title
        pub_date = None

    other_info = args.others

    markdown_line = '- [[%s](%s)] %s' % (pub_name, link_addr, paper_title)
    for key, item in other_info.items():
        markdown_line += ' [[%s](%s)]' % (key, item)
    markdown_line += '\n'

    '''Insert to README.md'''
    paper_list = open(args.list).read()

    # check if already existed
    title_search = re.search('-.*?' + paper_title + '.*?\n', paper_list)
    if title_search is not None:
        print('This paper is already in library.')
        sys.exit(0)

    # insert to paper list
    topic_search = re.search('## ' + topic_name + '.*?\n', paper_list)
    insert_point = topic_search.end(0)
    if pub_date is not None:
        if pub_date <= 2100:
            date_search = re.search('#### Before 2021.*?\n', paper_list[topic_search.end(0):])
            insert_point = topic_search.end(0) + date_search.end(0)

    new_paper_list = paper_list[:insert_point] + markdown_line + paper_list[insert_point:]

    with open(args.list, 'w') as file:
        file.write(new_paper_list)
