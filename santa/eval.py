""" Python version of the Santa 2014 Kaggle competition evaluation metric. """
__author__ = 'Joyce Noah-Vanhoucke'
__date__ = 'November 24, 2014'

import os
import csv
import time
import math
import datetime

from hours import Hours


class Toy:
    def __init__(self, toyid, arrival, duration):
        self.reference_start_time = datetime.datetime(2014, 1, 1, 0, 0)  # set when elf starts working on toy
        self.id = toyid
        self.arrival_minute = Hours.convert_to_minute(arrival)
        self.duration = int(duration)
        self.completed_minute = 0

    def outside_toy_start_period(self, start_minute):
        """ Checks that work on toy does not start outside of the allowed starting period.
        :param hrs: Hours class
        :param start_minute: minute the work is scheduled to start
        :return: True of outside of allowed starting period, False otherwise
        """
        return start_minute < self.arrival_minute

    def is_complete(self, start_minute, elf_duration, rating):
        """ Determines if the toy is completed given duration of work and elf's productivity rating
        :param start_minute: minute work started
        :param elf_duration: duration of work in minutes
        :param rating: elf's productivity rating
        :return: Boolean
        """
        if self.duration / rating <= elf_duration:
            self.completed_minute = start_minute + int(math.ceil(self.duration/rating))
            return True
        else:
            return False


class Elf:
    """ Each Elf starts with a rating of 1.0 and are available at 09:00 on Jan 1.  """
    def __init__(self, elfid):
        self.id = elfid
        self.rating = 1.0
        self.next_available_time = 540  # Santa's Workshop opens Jan 1, 2014 9:00 (= 540 minutes)
        self.rating_increase = 1.02
        self.rating_decrease = 0.90

    def update_elf(self, hrs, toy, start_minute, duration):
        """ Updates the elf's productivity rating and next available time based on last toy completed.
        :param hrs: Hours object for bookkeeping
        :param toy: Toy object for the toy the elf just finished
        :param start_minute: minute work started
        :param duration: duration of work, in minutes
        :return: void
        """
        self.update_next_available_minute(hrs, start_minute, duration)
        self.update_productivity(hrs, start_minute, int(math.ceil(toy.duration / self.rating)))

    def update_next_available_minute(self, hrs, start_minute, duration):
        """ Apply the resting time constraint and determine the next minute when the elf can work next.
        Here, elf can only start work during sanctioned times
        :param start_minute: time work started on last toy
        :param duration: duration of work on last toy
        :return: void
        """
        sanctioned, unsanctioned = hrs.get_sanctioned_breakdown(start_minute, duration)

        # enforce resting time based on the end_minute and the unsanctioned minutes that
        # need to be accounted for.
        end_minute = start_minute + duration
        if unsanctioned == 0:
            if hrs.is_sanctioned_time(end_minute):
                self.next_available_time = end_minute
            else:
                self.next_available_time = hrs.next_sanctioned_minute(end_minute)
        else:
            self.next_available_time = hrs.apply_resting_period(end_minute, unsanctioned)

    def update_productivity(self, hrs, start_minute, toy_required_minutes):
        """ Update the elf's productivity rating based on the number of minutes the toy required that were
        worked during sanctioned and unsanctioned times.
        max(0.5,
            min(2.0, previous_rating * (self.rating_increase ** sanctioned_hours) *
            (self.rating_decrease ** unsanctioned_hours)))
        :param hrs: hours object
        :param start_minute: minute work started
        :param toy_required_minutes: minutes required to build the toy (may be different from minutes elf worked)
        :return: void
        """
        # number of required minutes to build toy worked by elf, broken up by sanctioned and unsanctioned minutes
        sanctioned, unsanctioned = hrs.get_sanctioned_breakdown(start_minute, toy_required_minutes)
        self.rating = max(0.25,
                          min(4.0, self.rating * (self.rating_increase ** (sanctioned/60.0)) *
                              (self.rating_decrease ** (unsanctioned/60.0))))


def read_toys(toy_file, num_toys):
    """ Reads the toy file and returns a dictionary of Toys.
    Toy file format: ToyId, Arrival_time, Duration
        ToyId: toy id
        Arrival_time: time toy arrives. Format is: YYYY MM DD HH MM (space-separated)
        Duration: duration in minutes to build toy
    :param toy_file: toys input file
    :param hrs: hours object
    :param num_toys: total number of toys to build
    :return: Dictionary of toys
    """
    toy_dict = {}
    with open(toy_file, 'rb') as f:
        fcsv = csv.reader(f)
        fcsv.next()  # header row
        for row in fcsv:
            new_toy = Toy(row[0], row[1], row[2])
            toy_dict[new_toy.id] = new_toy
    if len(toy_dict) != num_toys:
        print '\n ** Read a file with {0} toys, expected {1} toys. Exiting.'.format(len(toy_dict), num_toys)
        exit(-1)
    return toy_dict


def score_submission(sub_file, myToys, hrs, NUM_TOYS, NUM_ELVES):
    """ Score the submission file, performing constraint checking. Returns the time (in minutes) when
    final present is complete.
    :param sub_file: submission file name. Headers: ToyId, ElfId, Start_Time, Duration
    :param myToys: toys dictionary
    :param hrs: hours object
    :return: time (in minutes) when final present is complete
    """

    myElves = {}
    complete_toys = []
    last_minute = 0
    row_count = 0
    with open(sub_file, 'rb') as f:
        fcsv = csv.reader(f)
        fcsv.next()  # header
        for row in fcsv:
            row_count += 1
            if row_count % 50000 == 0:
                print 'Starting toy: {0}'.format(row_count)

            current_toy = row[0]
            current_elf = int(row[1])
            start_minute = hrs.convert_to_minute(row[2])
            duration = int(row[3])

            if current_toy not in myToys:
                print 'Toy {0} not in toy dictionary.'.format(current_toy)
                exit(-1)
            elif myToys[current_toy].completed_minute > 0:
                print 'Toy {0} was completed at minute {1}'.format(current_toy, myToys[current_toy].completed_minute)
                exit(-1)

            if not 1 <= current_elf <= NUM_ELVES:
                print '\n ** Assigned elf does not exist: Elf {0}'.format(current_elf)
                exit(-1)

            # Create elf if this is the first time this elf is assigned a toy
            if current_elf not in myElves.keys():
                myElves[current_elf] = Elf(current_elf)

            # Check work starting constraints:
            # 1. within correct window of toy's arrival
            # 2. when elf is next allowed to work
            if myToys[current_toy].outside_toy_start_period(start_minute):
                print '\n ** Requesting work on Toy {0} at minute {1}: Work can start at {2} minutes'. \
                    format(current_toy, start_minute, myToys[current_toy].arrival_minute)
                exit(-1)
            if start_minute < myElves[current_elf].next_available_time:
                print '\n ** Elf {2} needs his rest, he is not available now ({0}) but will be later at {1}'. \
                    format(start_minute, myElves[current_elf].next_available_time, current_elf)
                exit(-1)

            # Check toy is complete based on duration and elf productivity
            if not myToys[current_toy].is_complete(start_minute, duration, myElves[current_elf].rating):
                print '\n ** Toy {0} is not complete'.format(current_toy)
                exit(-1)
            else:
                complete_toys.append(int(current_toy))
                if myToys[current_toy].completed_minute > last_minute:
                    last_minute = myToys[current_toy].completed_minute

            # Since toy started on time and is complete, update elf productivity
            myElves[current_elf].update_elf(hrs, myToys[current_toy], start_minute, duration)

    if len(complete_toys) != len(myToys):
        print "\n ** Not all toys are complete. Exiting."
        exit(-1)
    if max(complete_toys) != NUM_TOYS:
        print "\n ** max ToyId != NUM_TOYS."
        print "\n   max(complete_toys) = {0} versus NUM_TOYS = {1}".format(max(complete_toys), NUM_TOYS)
        exit(-1)
    else:
        score = last_minute * math.log(1.0 + len(myElves))
        print '\nSuccess!'
        print '  Score = {0}'.format(score)


# ======================================================================= #
# === MAIN === #

if __name__ == '__main__':
    """ Evaluation script for Helping Santa's Helpers, the 2014 Kaggle Holiday Optimization Competition. """
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

    start = time.time()

    myToys = read_toys(toy_file_path, num_toys)
    print ' -- All toys read. Starting to score submission. '
    hrs = Hours()
    score_submission(out_file_path, myToys, hrs, num_toys, num_elves)

    print 'total time = {0}'.format(time.time() - start)
