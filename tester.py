import autotest

class MyTest(autotest.AutoTestCase):

    def test1(self):
        self.addPreErrorMsg('pre test 1')
        self.addPostErrorMsg('post test 1')
        self.assertEqual(1, 1)


tester = MyTest()
tester.runTest('test1')