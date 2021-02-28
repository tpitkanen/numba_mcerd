import itertools
import unittest
import time

from numba_mcerd.timer import Timer, SplitTimer, TimerException


class TestTimer(unittest.TestCase):
    def test_running(self):
        timer = Timer()
        self.assertFalse(timer.is_running)
        timer.start()
        self.assertTrue(timer.is_running)
        timer.stop()
        self.assertFalse(timer.is_running)

    def test_init_and_start_running(self):
        timer = Timer.init_and_start()
        self.assertTrue(timer.is_running)
        timer.stop()
        self.assertFalse(timer.is_running)

    def test_exception(self):
        timer = Timer.init_and_start()
        with self.assertRaises(TimerException) as cm:
            timer.start()
        self.assertEqual(Timer.MSG_TIMER_RUNNING, str(cm.exception))

        timer2 = Timer.init_and_start()
        timer2.stop()
        with self.assertRaises(TimerException) as cm:
            timer2.stop()
        self.assertEqual(Timer.MSG_TIMER_FINISHED, str(cm.exception))

        timer3 = Timer()
        with self.assertRaises(TimerException) as cm:
            timer3.stop()
        self.assertEqual(Timer.MSG_TIMER_NOT_STARTED, str(cm.exception))

        timer4 = Timer.init_and_start()
        timer4.stop()
        with self.assertRaises(TimerException) as cm:
            timer4.start()
        self.assertEqual(Timer.MSG_TIMER_FINISHED, str(cm.exception))

        timer5 = Timer.init_and_start()
        timer5.stop()
        with self.assertRaises(TimerException) as cm:
            timer5.stop()
        self.assertEqual(Timer.MSG_TIMER_FINISHED, str(cm.exception))

    def test_elapsed(self):
        """This is a timing-based test. Slow execution may cause failure."""
        time.sleep(0.1)
        timer = Timer()
        self.assertEqual(0.0, timer.elapsed)
        timer.start()
        time.sleep(0.2)
        self.assertAlmostEqual(0.2, timer.elapsed, places=2)
        timer.stop()
        self.assertAlmostEqual(0.2, timer.elapsed, places=2)

    def test_str(self):
        timer = Timer()
        self.assertEqual("Timer (start=None stop=None elapsed=0.0)", str(timer))


class TestSplitTimer(unittest.TestCase):
    def test_split(self):
        """This is a timing-based test. Slow execution may cause failure."""
        timer = SplitTimer()
        self.assertEqual([], timer.splits)
        timer.start()
        time.sleep(0.1)
        self.assertAlmostEqual(0.1, timer.split(), places=1)
        time.sleep(0.2)
        self.assertAlmostEqual(0.3, timer.split(), places=1)
        time.sleep(0.3)
        self.assertAlmostEqual(0.6, timer.elapsed, places=1)
        timer.stop()

        for target_time, split_time in itertools.zip_longest([0.1, 0.3, 0.6], timer.elapsed_splits, fillvalue=None):
            self.assertAlmostEqual(target_time, split_time, places=1)

        for target_time, lap_time in itertools.zip_longest([0.1, 0.2, 0.3], timer.elapsed_laps, fillvalue=None):
            self.assertAlmostEqual(target_time, lap_time, places=1)

    def test_str(self):
        timer = SplitTimer()
        self.assertEqual("SplitTimer (start=None stop=None elapsed=0.0 splits=[])", str(timer))
