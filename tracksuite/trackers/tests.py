import unittest
from unittest import TestCase
from trackers.siamfc import load_video

class TrackerTest(TestCase):
    
    def test_simple(self):
        assert 2 == 2

if __name__ == "__main__":
    unittest.main(warnings='ignore')