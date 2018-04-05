from unittest import TextTestRunner, TestLoader
from unittest.result import TestResult
from unittest.signals import registerResult
from autotest.AutoTestCase import Watcher
import json
import unittest

class GraderResult(TestResult):

    def __init__(self, *args, **kwargs):
        TestResult.__init__(self, *args, **kwargs)

    def printErrors(self):
        pass

    def printErrorList(self, flavour, errors):
        pass


class GraderRunner(TextTestRunner):

    resultclass = GraderResult

    def __index__(self):
        TextTestRunner.__init__(self)
        self.verbosity = 0

    def run(self, test):
        "Run the given test case or test suite."
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer

        startTestRun = getattr(result, 'startTestRun', None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, 'stopTestRun', None)
            if stopTestRun is not None:
                stopTestRun()
            result.testsRun
        return result

#
# def main(case=None, traceback=False):
#     # if not case:
#     #     case = Watcher.instance
#     #     if not case:
#     #         res = {'score': 1, 'output': []}
#     #         print(json.dumps(res))
#     #         return
#     case = Watcher.instance
#     # print(TestLoader().)
#     suite = TestLoader().loadTestsFromTestCase(case)
#     testResult = GraderRunner().run(suite)
#     total = testResult.testsRun
#     if total == 0:
#         res = {'score': 1, 'output': []}
#     else:
#         errors = [x[1] for x in testResult.errors]
#         failures = [x[1] for x in testResult.failures]
#         score = 1 - 1.0 * (len(errors) + len(failures)) / total
#         res = {'score': score, 'test_output': []}
#         for i in errors + failures:
#             if not traceback:
#                 index = i.find('AutoTestCaseError:')
#                 if index > -1:
#                     i = i[index:]
#             res['test_output'].append({'type': 'error', 'string': i})
#     print(json.dumps(res))


