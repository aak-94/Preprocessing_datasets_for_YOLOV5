import unittest
import sys
sys.path.append('g:\\Complete_Projects\\Projects_list\\06_Preprocessing_datasets_for_YOLOV5\\')
from utils import xml_to_yolo 


class TestCommon(unittest.TestCase):

    def setUp(self):
        self.values=['label',100, 200, 150, 400]
        self.img_size=600
        self.decimal_limit=6
        self.required_results=(0.208333, 0.5, 0.083333, 0.333333)

    def test_dict_to_yolo_label(self):
        
        results=xml_to_yolo(self.values,self.img_size,self.decimal_limit)
        self.assertEqual(results,self.required_results),'Test: test_dict_to_yolo_label failed'


if __name__=='__main__':
    unittest.main()