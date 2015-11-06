# coding: utf8

import datasets
import datasets.imagenet_ilsvrc
import os
import datasets.imdb_imagenet
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class imagenet_ilsvrc(datasets.imdb_imagenet):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb_imagenet.__init__(self, 'ILSVRC_' + year + '_DET_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'ILSVRC' + self._year)
        self._classes = self._load_imagenet_label2words()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index() # ILSVRC2013_train_extra0/ILSVRC2013_train_00000001
        self._image_sizes = self._load_image_sizes()
        self._remove_negative_data()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'ILSVRCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

        print 'Number of classes:', len(self._classes)
        print 'Number of images:', len(self._image_index)

    def _load_imagenet_label2words(self):
        filename = os.path.join(self._data_path, 'devkit', 'data', 'map_det.txt')
        assert os.path.exists(filename), \
               'label2words not found at: {}'.format(filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        classes = tuple([l.split(' ')[0] for l in lines])
        result = ('__background__',) + classes
        return result

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', 'DET',
                                  self._image_set, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET',
                                     self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.split(' ')[0] for x in f.readlines()]
        return image_index

    def _load_image_sizes(self):
        filename = os.path.join(self._devkit_path, 'image_sizes_' + self._image_set + '.txt')
        assert os.path.exists(filename), \
               'image_sizes_{}.txt not found at: {}'.format(self._image_set, filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        sizes = []
        for l in lines:
            sp = l.split(' ')
            sizes.append((int(sp[0]), int(sp[1])))
        return sizes

    def _remove_negative_data(self):
        filename = os.path.join(self._devkit_path, self.name + '_positive_data')
        if not os.path.exists(filename):
            print 'No positive data list found at {}, treating all data as positive'.format(filename)
            return

        print 'Removing negative data according to {}'.format(filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        positive_data_indices = [int(l.split(' ')[0]) for l in lines]
        self._image_index = [self._image_index[x] for x in positive_data_indices]
        self._image_sizes = [self._image_sizes[x] for x in positive_data_indices]

    def _get_default_path(self):
        """
        Return the default path where IMAGENET ILSVRC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ILSVRCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        print 'Loading ground-truth RoI DB'
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = []
        for i, index in enumerate(self.image_index):
            anno = self._load_imagenet_annotation(index)
            gt_roidb.append(anno)
            if (i+1) % 5000 == 0:
                print '{}'.format(i+1)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2015 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2015 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the IMAGENET ILSVRC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        # readme.txt:305
        # Images without any annotated objects may not have a corresponding xml file.
        if not os.path.exists(filename):
            data = minidom.parseString('<useless></useless>')
        else:
            with open(filename) as f:
                data = minidom.parseString(f.read())

        # some images don't have objects even though they have .xml!!!!!
        objs = data.getElementsByTagName('object')
        if index == 'ILSVRC2014_train_0006/ILSVRC2014_train_00060036':
            # the second object seems wrong
            # <xmin>1</xmin>
            # <xmax>0</xmax>
            # <ymin>498</ymin>
            # <ymax>498</ymax>
            objs = [objs[0]]
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based.
            # 有些标注本来就是0，但是又有的x/ymax和宽/高一样，所以原始信息应该是0/1混合based
            x1 = max(0, float(get_data_from_tag(obj, 'xmin')) - 1)
            y1 = max(0, float(get_data_from_tag(obj, 'ymin')) - 1)
            x2 = max(0, float(get_data_from_tag(obj, 'xmax')) - 1)
            y2 = max(0, float(get_data_from_tag(obj, 'ymax')) - 1)
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_ilsvrc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # ILSVRCdevkit/results/ILSVRC2015/Main/comp4-44503_det_test_accordion.txt
        path = os.path.join(self._devkit_path, 'results', 'ILSVRC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} ILSVRC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the ILSVRCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_ilsvrc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.ilsvrc_imagenet('trainval', '2015')
    res = d.roidb
    from IPython import embed; embed()
