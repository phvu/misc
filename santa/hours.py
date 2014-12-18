import datetime


class Hours:
    """ Hours class takes care of time accounting. Note that convention is
     9:00-9:05 is work taking place on the 00, 01, 02, 03, 04 minutes, so 5 minutes of work.
     Elf is available for next work at 9:05
     Class members day_start, day_end are in minutes relative to the day
    """
    def __init__(self):
        self.hours_per_day = 10  # 10 hour day: 9 - 19
        self.day_start = 9 * 60
        self.day_end = (9 + self.hours_per_day) * 60
        self.reference_start_time = datetime.datetime(2014, 1, 1, 0, 0)
        self.minutes_in_24h = 24 * 60

    @staticmethod
    def convert_to_minute(arrival):
        """ Converts the arrival time string to minutes since the reference start time,
        Jan 1, 2014 at 00:00 (aka, midnight Dec 31, 2013)
        :param arrival: string in format '2014 12 17 7 03' for Dec 17, 2014 at 7:03 am
        :return: integer (minutes since arrival time)
        """
        time = arrival.split(' ')
        dd = datetime.datetime(int(time[0]), int(time[1]), int(time[2]), int(time[3]), int(time[4]))
        age = dd - datetime.datetime(2014, 1, 1, 0, 0)
        return int(age.total_seconds() / 60)

    @staticmethod
    def convert_to_datetime(minute):
        return datetime.datetime(2014, 1, 1, 0, 0) + datetime.timedelta(minutes=minute)

    def is_sanctioned_time(self, minute):
        """ Return boolean True or False if a given time (in minutes) is a sanctioned working day minute.  """
        return ((minute - self.day_start) % self.minutes_in_24h) < (self.hours_per_day * 60)

    def get_sanctioned_breakdown(self, start_minute, duration):
        """ Whole days (24-hr time periods) contribute fixed quantities of sanctioned and unsanctioned time. After
        accounting for the whole days in the duration, the remainder minutes are tabulated as un/sanctioned.
        :param start_minute:
        :param duration:
        :return:
        """
        if self.is_sanctioned_time(start_minute):
            sanctioned_start_minute = start_minute
            unsanctioned = 0
        else:
            sanctioned_start_minute = self.next_sanctioned_minute(start_minute)
            unsanctioned = sanctioned_start_minute - start_minute
            duration -= unsanctioned

        full_days = duration / (self.minutes_in_24h)
        sanctioned = full_days * self.hours_per_day * 60
        unsanctioned += full_days * (24 - self.hours_per_day) * 60
        remainder_start = sanctioned_start_minute + full_days * self.minutes_in_24h
        for minute in xrange(remainder_start, sanctioned_start_minute+duration):
            if self.is_sanctioned_time(minute):
                sanctioned += 1
            else:
                unsanctioned += 1
        return sanctioned, unsanctioned

    def next_sanctioned_minute(self, minute):
        """ Given a minute, finds the next sanctioned minute.
        :param minute: integer representing a minute since reference time
        :return: next sanctioned minute
        """
        # next minute is a sanctioned minute
        if self.is_sanctioned_time(minute) and self.is_sanctioned_time(minute+1):
            return minute + 1
        num_days = minute / self.minutes_in_24h

        added_day = 1
        if (minute - num_days*(self.minutes_in_24h)) < self.day_start:
            added_day = 0
        return self.day_start + (num_days + added_day) * self.minutes_in_24h

    def apply_resting_period(self, start, num_unsanctioned):
        """ Enforces the rest period and returns the minute when the elf is next available for work.
        Rest period is applied during sanctioned hours, so 5 hours rest period requires 5 hours sanctioned.
        As a result, rest periods will end during the middle of sanctioned hours or at exactly the end of
        sanctioned hours. If the rest period ends exactly when the rest period ends, this returns the next
        morning at 9:00 am as the next available minute.
        :param start: minute the REST period starts
        :param num_unsanctioned: always > 0 number of unsanctioned minutes that need resting minutes
        :return: next available minute after rest period has been applied. If rest period ends with the working day
                 returns the next morning at 9:00 am
        """
        num_days_since_jan1 = start / self.minutes_in_24h
        rest_time = num_unsanctioned
        rest_time_in_working_days = rest_time / (60 * self.hours_per_day)
        rest_time_remaining_minutes = rest_time % (60 * self.hours_per_day)

        # rest time is only applied to sanctioned work hours. If local_start is at an unsanctioned time,
        # need to set it to be the next start of day
        local_start = start % self.minutes_in_24h  # minute of the day (relative to a current day) the work starts
        if local_start < self.day_start:
            local_start = self.day_start
        elif local_start > self.day_end:
            num_days_since_jan1 += 1
            local_start = self.day_start

        if local_start + rest_time_remaining_minutes > self.day_end:
            rest_time_in_working_days += 1
            rest_time_remaining_minutes -= (self.day_end - local_start)
            local_start = self.day_start

        total_days = num_days_since_jan1 + rest_time_in_working_days
        return total_days * self.minutes_in_24h + local_start + rest_time_remaining_minutes