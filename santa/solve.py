import csv
import random
from models import Toy, Elves
from hours import Hours
import cPickle
import uuid
import glob


def read_toys(toy_file, num_toys):
    """ Reads the toy file and returns a dictionary of Toys.
    Toy file format: ToyId, Arrival_time, Duration
        ToyId: toy id
        Arrival_time: time toy arrives. Format is: YYYY MM DD HH MM (space-separated)
        Duration: duration in minutes to build toy
    :param toy_file: toys input file
    :param num_toys: total number of toys to build
    :return: Dictionary of toys
    """
    toys = []
    with open(toy_file, 'rb') as f:
        fcsv = csv.reader(f)
        fcsv.next()  # header row
        for row in fcsv:
            toys.append(Toy(row[0], row[1], row[2]))
    if len(toys) != num_toys:
        print '\n ** Read a file with {0} toys, expected {1} toys. Exiting.'.format(len(toys), num_toys)
        exit(-1)
    return toys


#######################################################################################################################
# http://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):

    def _get_daemon(self):
        """
        make 'daemon' attribute always return False
        """
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    """
    We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    because the latter is only a wrapper function, not a proper class.
    """
    Process = NoDaemonProcess


#######################################################################################################################

TOY_OBJ_FILE = './tmp/toys.obj'


def assign_elves_process(process_data):
    (toy_indices, elves_file, hrs, recursive_level) = process_data

    if len(toy_indices) == 1:
        # toy = toy_indices[0]
        # assert type(toy) is Toy
        with open(TOY_OBJ_FILE, 'r') as f:
                toys = cPickle.load(f)
        toy = toys[toy_indices[0]]
        with open(elves_file, 'r') as f:
            elves = cPickle.load(f)

        elves2 = elves.assign(toy, hrs)

        out_file = os.path.join(os.path.split(elves_file)[0], uuid.uuid4().hex + '.pkl')
        while os.path.exists(out_file):
            out_file = os.path.join(os.path.split(elves_file)[0], uuid.uuid4().hex + '.pkl')

        with open(out_file, 'w') as f:
            cPickle.dump(elves2, f, cPickle.HIGHEST_PROTOCOL)
        del toys
        del elves
        del elves2
        return out_file
    else:
        half = len(toy_indices)/2

        if recursive_level < 2:
            pool = MyPool(processes=2)
            elves1_file, elves3_file = pool.map(assign_elves_process,
                                                [(toy_indices[:half], elves_file, hrs, recursive_level+1),
                                                 (toy_indices[half:], elves_file, hrs, recursive_level+1)])

            elves2_file, elves4_file = pool.map(assign_elves_process,
                                                [(toy_indices[half:], elves1_file, hrs, recursive_level+1),
                                                 (toy_indices[:half], elves3_file, hrs, recursive_level+1)])
            pool.close()
            pool.join()
        else:
            elves1_file = assign_elves_process((toy_indices[:half], elves_file, hrs, recursive_level+1))
            elves2_file = assign_elves_process((toy_indices[half:], elves1_file, hrs, recursive_level+1))

            elves3_file = assign_elves_process((toy_indices[half:], elves_file, hrs, recursive_level+1))
            elves4_file = assign_elves_process((toy_indices[:half], elves3_file, hrs, recursive_level+1))

        with open(elves2_file, 'r') as f:
            elves2 = cPickle.load(f)
        with open(elves4_file, 'r') as f:
            elves4 = cPickle.load(f)
        elves2_better = elves2.cost() < elves4.cost()
        del elves2
        del elves4

        os.remove(elves1_file)
        os.remove(elves3_file)
        if elves2_better:
            os.remove(elves4_file)
            return elves2_file
        else:
            os.remove(elves2_file)
            return elves4_file


def assign_elves_single_process(toy_indices, elves_file, hrs):
    if len(toy_indices) == 1:
        idx = toy_indices[0]
        with open(TOY_OBJ_FILE, 'r') as f:
            toys = cPickle.load(f)
        with open(elves_file, 'r') as f:
            elves = cPickle.load(f)
        elves2 = elves.assign(toys[idx], hrs)

        out_file = os.path.join(os.path.split(elves_file)[0], uuid.uuid4().hex + '.pkl')
        while os.path.exists(out_file):
            out_file = os.path.join(os.path.split(elves_file)[0], uuid.uuid4().hex + '.pkl')

        with open(out_file, 'w') as f:
            cPickle.dump(elves2, f, cPickle.HIGHEST_PROTOCOL)
        del toys
        del elves
        del elves2
        return out_file
    else:
        half = len(toy_indices)/2

        elves1_file = assign_elves_single_process(toy_indices[:half], elves_file, hrs)
        elves2_file = assign_elves_single_process(toy_indices[half:], elves1_file, hrs)

        elves3_file = assign_elves_single_process(toy_indices[half:], elves_file, hrs)
        elves4_file = assign_elves_single_process(toy_indices[:half], elves3_file, hrs)

        with open(elves2_file, 'r') as f:
            elves2 = cPickle.load(f)
        with open(elves4_file, 'r') as f:
            elves4 = cPickle.load(f)
        elves2_better = elves2.cost() < elves4.cost()
        del elves2
        del elves4
        return elves2_file if elves2_better else elves4_file


def assign_elves(toys, elves, hrs):
    return assign_elves_process((toys, elves, hrs, 0))
    # return assign_elves_single_process(toys, elves, hrs)


def solve(toy_file, num_toys, num_elves, out_file):

    all_toys = read_toys(toy_file, num_toys)
    with open(TOY_OBJ_FILE, 'w') as f:
        cPickle.dump(all_toys, f, cPickle.HIGHEST_PROTOCOL)
    hrs = Hours()

    """
    all_toys_idx = range(0, len(all_toys))
    del all_toys

    elves_init = Elves(num_elves)
    elves_init_file = './tmp/elves_init.obj'
    with open(elves_init_file, 'w') as f:
        cPickle.dump(elves_init, f, cPickle.HIGHEST_PROTOCOL)
    del elves_init

    elves_best = None
    for i in range(0, 1):
        for s in glob.glob('./tmp/*.pkl'):
            os.remove(s)
        elves_file = assign_elves(all_toys_idx, elves_init_file, hrs)
        with open(elves_file, 'r') as f:
            elves = cPickle.load(f)
        if elves_best is None or elves_best.cost() > elves.cost():
            elves_best = elves
        random.shuffle(all_toys_idx)

    """
    toy_groups = {}
    for toy_idx, toy in enumerate(all_toys):
        if toy.arrival_minute not in toy_groups:
            toy_groups[toy.arrival_minute] = []
        toy_groups[toy.arrival_minute].append(toy_idx)

    del all_toys

    toy_groups = [(arrival_minute, toys) for arrival_minute, toys in toy_groups.iteritems()]

    elves_best = None
    for i in range(0, min(len(toy_groups), 1)):
        random.shuffle(toy_groups)

        elves_init = Elves(num_elves)
        elves_init_file = './tmp/elves_init.obj'
        with open(elves_init_file, 'w') as f:
            cPickle.dump(elves_init, f, cPickle.HIGHEST_PROTOCOL)
        del elves_init

        elves_group_file = elves_init_file
        for grp_idx, (arrival_minute, toys) in enumerate(toy_groups):
            print 'Working on group %d/%d' % (grp_idx, len(toy_groups))

            # find the "best" way to assign this group (toys)
            elves_group_best_file = None
            elves_group_best_score = None
            for j in range(0, min(len(toys), 1)):
                random.shuffle(toys)
                elves_new_file = assign_elves(toys, elves_group_file, hrs)
                with open(elves_new_file, 'r') as f:
                    elves_new = cPickle.load(f)
                if elves_group_best_score is None or elves_group_best_score > elves_new.cost():
                    elves_group_best_score = elves_new.cost()
                    elves_group_best_file = elves_new_file

            elves_group_file = elves_group_best_file
        with open(elves_group_file, 'r') as f:
            elves = cPickle.load(f)
        if elves_best is None or elves_best.cost() > elves.cost():
            elves_best = elves

    elves_best.write_jobs(out_file)
    print 'Cost:', elves_best.cost()


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) == 2:
        toy_file_path = os.path.join(os.getcwd(), 'data/toys_rev2.csv')
        out_file_path = sys.argv[1]
        num_toys = 10000000
        num_elves = 900
    else:
        toy_file_path = sys.argv[1]
        out_file_path = sys.argv[2]
        num_toys = int(sys.argv[3])
        num_elves = int(sys.argv[4])

    import time
    t = time.time()
    solve(toy_file_path, num_toys, num_elves, out_file_path)
    t = time.time() - t
    print 'Total time:', t