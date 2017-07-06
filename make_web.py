from util import html
from data import image_folder
from collections import defaultdict, OrderedDict
from itertools import product, chain
import string
import os
import shutil
import ntpath

ds = 'artists2photo'

methods = defaultdict(OrderedDict)
methods[('A', 'B', 'C')]['arts_paired_23'] = 'Cycles of lengths 2 and 3'
methods[('A', 'B', 'C')]['arts_paired_235'] = 'Cycles of lengths 2, 3 and 5'
methods[('B', 'A')]['monet2photo_baseline'] = 'monet2photo CycleGAN baseline'
methods[('C', 'A')]['cezanne2photo_baseline'] = 'cezanne2photo CycleGAN baseline'
methods[('B', 'C')]['monet2cezanne_baseline'] = 'monet2cezanne CycleGAN baseline'

sets = {'A': 'real photos', 'B': 'Monet paintings', 'C': 'Cezanne paintings'}
phases = ['train', 'test']
results_dir = './results'
web_dir = './webpage'
win_size = 256

def local_set_name(s, cover):
    return string.ascii_uppercase[cover.index(s)]

shutil.rmtree(web_dir)

webpage_home = html.HTML(web_dir, '')

# copy ims
for meth in chain(*methods.values()):
    shutil.copytree(
        os.path.join(results_dir, meth),
        os.path.join(webpage_home.get_image_dir(), meth),
    )
    print('finished copying %s results' % meth)

print('finished copying all results')
print()

def meth_imgs_dataset(meth, phase):
    return image_folder.make_dataset( \
        os.path.join(webpage_home.get_image_dir(), meth, '%s_latest' % phase, 'images'))

def real_imgs_dir(s, phase):
    return os.path.join(webpage_home.get_image_dir(), '%s_%s' % (s, phase))

# find shared reals
reals = {}

for s in sets:
    for phase in phases:
        reals_s = defaultdict(lambda: [None, 0]) # set to path
        max_meths = 0
        for cover, meths in methods.items():
            if s not in cover:
                continue
            max_meths += len(meths)
            ms = local_set_name(s, cover)
            suff = '_real_%s' % ms
            for meth in meths:
                for im in meth_imgs_dataset(meth, phase):
                    short_path = ntpath.basename(im)
                    name = os.path.splitext(short_path)[0]
                    if name.endswith(suff):
                        pair = reals_s[name[:-len(suff)]]
                        pair[0] = im
                        pair[1] += 1
        reals[(s, phase)] = {name: pair[0] for name, pair in reals_s.items() if pair[1] == max_meths}


for (s, phase), ims in reals.items():
    target_dir = real_imgs_dir(s, phase)
    os.makedirs(target_dir)
    for im, path in ims.items():
        shutil.copy2(path, os.path.join(target_dir, '%s.png' % im))

print('finished collecting real images:')
for k, v in reals.items():
    print('%s:\t%d images' % (str(k), len(v)))
print()

for meth in chain(*methods.values()):
    for phase in phases:
        for im in image_folder.make_dataset(os.path.join(webpage_home.get_image_dir(), meth, '%s_latest' % phase, 'images')):
            short_path = ntpath.basename(im)
            name = os.path.splitext(short_path)[0]
            if 'fake' not in name:
                os.remove(im)

print('finished removing irrelevant images')
print()

webpages = {}
for s, phase in product(sets, phases):
    webpage = html.HTML(
        web_dir,
        'Compare %s' % ', '.join(chain(*methods.values())),
        html_name = '%s_%s' % (s, phase)
    )

    webpages[(s, phase)] = webpage

    real_im_base_dir = os.path.join('%s_%s' % (s, phase))

    for im_name in reals[(s, phase)]:
        im_path = os.path.join(real_im_base_dir, '%s.png' % im_name)

        webpage.add_header('%s\t%s\t%s' % (s, phase, im_name))

        ims = []
        txts = []
        links = []

        ims.append(im_path)
        txts.append('real %s' % im_name)
        links.append(im_path)

        for ss in sets:
            if ss == s:
                continue
            for cover in methods:
                if s not in cover or ss not in cover:
                    continue
                ms = local_set_name(s, cover)
                mss = local_set_name(ss, cover)
                if len(cover) > 2:
                    fake_im = '%s_fake_%s%s.png' % (im_name, ms, mss)
                else:
                    fake_im = '%s_fake_%s.png' % (im_name, mss)
                for meth in methods[cover]:
                    im_path = os.path.join(meth, '%s_latest' % phase, 'images',
                        fake_im)
                    ims.append(im_path)
                    txts.append('%s: %s => %s' % (meth, s, ss))
                    links.append(im_path)

        webpage.add_images(ims, txts, links, width=win_size)

    webpage.save()

    print('finished building webpage for set %s phase %s' % (s, phase))

print()

import dominate.tags as tags

with webpage_home.doc:
    tags.h1('Methods:')
    with tags.div().add(tags.ol()):
        for meth, desc in chain(*(d.items() for d in methods.values())):
            tags.li(tags.h3('%s:\t%s' % (meth, desc)))

    tags.h1('Datasets:')
    with tags.div().add(tags.ol()):
        for s, desc in sets.items():
            tags.h2('Dataset %s:\t%s' % (s, desc))
            with tags.div().add(tags.ol()):
                for phase in phases:
                    href = '%s.html' % webpages[(s, phase)].html_name
                    tags.li(tags.a('Translate from %s in %s phase' % (s, phase), href = href))

webpage_home.save()
print('home page done')

