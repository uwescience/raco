# -*- coding: UTF-8 -*-
import raco.myrial.myrial_test as myrial_test
from raco.fake_data import FakeData


class TestSamplingOperations(myrial_test.MyrialTestCase, FakeData):
    def setUp(self):
        super(TestSamplingOperations, self).setUp()

        self.db.ingest(TestSamplingOperations.emp_key,
                       TestSamplingOperations.emp_table,
                       TestSamplingOperations.emp_schema)

    def run_samplescan(self, sample_size, sample_type):
        query = """
        emp = SAMPLESCAN({rel_key}, {size}, {type});
        STORE(emp, OUTPUT);
        """.format(rel_key=self.emp_key, size=sample_size, type=sample_type)

        res = self.execute_query(query)
        self.assertEquals(len(res), sample_size)

    def test_samplescan__wr_zero(self):
        self.run_samplescan(0, 'WR')

    def test_samplescan__wor_zero(self):
        self.run_samplescan(0, 'WoR')

    def test_samplescan__wr_one(self):
        self.run_samplescan(1, 'WR')

    def test_samplescan__wor_one(self):
        self.run_samplescan(1, 'WoR')

    def test_samplescan__wr_all(self):
        self.run_samplescan(len(self.emp_table), 'WR')

    def test_samplescan__wor_all(self):
        self.run_samplescan(len(self.emp_table), 'WoR')
