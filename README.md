# pyautotest


## Example

```
import autotest

class MyTest(autotest.AutoTestCase):

    def first_test(self):
        self.addPreErrorMsg('pre test 1')
        self.addPostErrorMsg('post test 1')
        self.assertEqual(1, 2)
        
    def another_test(self):
        self.addPreErrorMsg('pre test 2')
        self.addPostErrorMsg('post test 2')
        self.assertEqual(2, 4)


tester = MyTest()
tester.runTest('first_test', 'another_test')
```
