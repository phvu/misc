from hours import Hours
import math


class Toy(object):
    def __init__(self, toy_id, arrival, duration):
        self.id = int(toy_id)
        self.arrival_minute = Hours.convert_to_minute(arrival)
        self.duration = int(duration)


class Elf(object):
    def __init__(self, elf_id):
        self.id = elf_id
        self.rating = 1.0
        self.next_start_time = 0
        self.rating_increase = 1.02
        self.rating_decrease = 0.90

    def copy(self):
        e = Elf(self.id)
        e.rating = self.rating
        e.next_start_time = self.next_start_time
        return e

    def update(self, next_start_time, sanctioned, unsanctioned):
        self.next_start_time = next_start_time
        self.rating = max(0.25,
                          min(4.0, self.rating * (self.rating_increase ** (sanctioned/60.0)) *
                              (self.rating_decrease ** (unsanctioned/60.0))))


class Elves(object):
    def __init__(self, num_elves):
        self.num_elves = num_elves
        self.elves = []
        self.jobs = []
        self.finish_time = 0

    def copy(self):
        elves = Elves(self.num_elves)
        elves.elves = [e.copy() for e in self.elves]
        elves.jobs = [data[:] for data in self.jobs]
        elves.finish_time = self.finish_time
        return elves

    def cost(self):
        return self.finish_time * math.log(1.0 + len(self.elves))

    @staticmethod
    def assign_elf(current_elf, toy, input_time, hrs):
        # double checks that work starts during sanctioned work hours
        if hrs.is_sanctioned_time(input_time):
            start_time = input_time
        else:
            start_time = hrs.next_sanctioned_minute(input_time)
        # start_time = input_time

        duration = int(math.ceil(toy.duration / current_elf.rating))
        sanctioned, unsanctioned = hrs.get_sanctioned_breakdown(start_time, duration)

        if unsanctioned == 0:
            return hrs.next_sanctioned_minute(start_time + duration - 1), start_time, duration
        else:
            return hrs.apply_resting_period(start_time + duration - 1, unsanctioned), start_time, duration

    def write_jobs(self, out_file):
        jobs = sorted(self.jobs, key=lambda x: x[2])
        with open(out_file, 'w') as f:
            f.write('ToyId,ElfId,StartTime,Duration\n')
            for toy_id, elf_id, start_minute, duration in jobs:
                start_time = Hours.convert_to_datetime(start_minute)
                f.write('%d,%d,%d %d %d %d %d,%d\n' % (toy_id, elf_id,
                                                       start_time.year, start_time.month,
                                                       start_time.day, start_time.hour,
                                                       start_time.minute, duration))

    def assign(self, toy, hrs):
        """
        Select the best elf for this toy such that it minimizes the cost
        :param toy:
        :return:
        """
        assert type(toy) is Toy

        elves = self.copy()

        assert len(elves.elves) < elves.num_elves or len(elves.elves) > 0

        new_elf_obj = float("inf")

        if len(elves.elves) < elves.num_elves:
            new_elf = Elf(len(elves.elves)+1)
            (new_elf_next_start_time, new_elf_start_time,
             new_elf_duration) = Elves.assign_elf(new_elf, toy, toy.arrival_minute, hrs)
            new_elf_obj = max(elves.finish_time, (new_elf_start_time + new_elf_duration)) * math.log(1.0 + len(elves.elves) + 1.0)

        old_elf_obj = float("inf")
        if len(elves.elves) > 0:
            old_elf = None
            old_elf_next_start_time = None
            old_elf_start_time = None
            old_elf_duration = None
            for elf in elves.elves:
                (elf_next_start_time, elf_start_time,
                 elf_duration) = Elves.assign_elf(elf, toy, elf.next_start_time, hrs)
                obj = max(elves.finish_time, (elf_start_time + elf_duration)) * math.log(1.0 + len(elves.elves))
                if old_elf_obj > obj:
                    old_elf_obj = obj
                    old_elf_next_start_time = elf_next_start_time
                    old_elf = elf
                    old_elf_start_time = elf_start_time
                    old_elf_duration = elf_duration

        if new_elf_obj < old_elf_obj:
            # use new elf
            sanctioned, unsanctioned = hrs.get_sanctioned_breakdown(new_elf_start_time, new_elf_duration)
            new_elf.update(new_elf_next_start_time, sanctioned, unsanctioned)
            elves.jobs.append([toy.id, new_elf.id, new_elf_start_time, new_elf_duration])
            elves.elves.append(new_elf)
            elves.finish_time = max(elves.finish_time, new_elf_start_time + new_elf_duration)
        else:
            # use old elf
            sanctioned, unsanctioned = hrs.get_sanctioned_breakdown(old_elf_start_time, old_elf_duration)
            old_elf.update(old_elf_next_start_time, sanctioned, unsanctioned)
            elves.jobs.append([toy.id, old_elf.id, old_elf_start_time, old_elf_duration])
            elves.finish_time = max(elves.finish_time, old_elf_start_time + old_elf_duration)

        return elves

