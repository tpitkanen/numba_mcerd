import unittest
import time

from numba_mcerd.timer import Timer, SplitTimer, TimerException


class TestTimer(unittest.TestCase):
    def test_running(self):
        timer = Timer()
        self.assertFalse(timer.running)
        timer.start()
        self.assertTrue(timer.running)
        timer.stop()
        self.assertFalse(timer.running)

    def test_init_and_start_running(self):
        timer = Timer.init_and_start()
        self.assertTrue(timer.running)
        timer.stop()
        self.assertFalse(timer.running)

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
        timer = Timer()
        self.assertEqual(0.0, timer.elapsed)
        timer.start()
        time.sleep(1.5)
        self.assertAlmostEqual(1.5, timer.elapsed, places=2)
        timer.stop()
        self.assertAlmostEqual(1.5, timer.elapsed, places=2)


class TestSplitTimer(unittest.TestCase):
    pass  # TODO
