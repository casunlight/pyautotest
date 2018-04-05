import unittest
import numpy as np
import types, py_compile
from pandas import Series, DataFrame
from subprocess import check_output, CalledProcessError
from six import with_metaclass
import sys

#
# if sys.version_info[0] == 2: # python2
#     from exceptions import Exception


def RaiseError(func):
    def wrapped(*args, **kwargs):
        ans = func(*args, **kwargs)
        error = args[0].errorHandler.errorBuffer.OutputErrors(clear=not args[0].stickyBuffer)
        if not ans:
            args[0].stickyBuffer = False
            args[0].errorHandler.Clear()
            raise AutoTestCaseError(error)
    return (wrapped)


def ForbidCalling(ans):
    """
    forbid to call the method
    """
    raise AttributeError("forbid to call the method")


class AutoTestCaseError(Exception):
    pass


class Watcher(type):

    instance = None

    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            Watcher.instance = cls  # save the last subclass of AutoTestCase
        super(Watcher, cls).__init__(name, bases, clsdict)


class AutoTestCase(with_metaclass(Watcher, unittest.TestCase)):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.errorHandler = ErrorHandler()
        self.stickyBuffer = False

    def runTest(self, *args):
        for test in args:
            getattr(self, test)()
        else:
            return 'AutoTest cases passed: {}'.format(len(args))

    #    assertIn = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertIs = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertDictEqual = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertSetEqual  = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertTupleEqual= property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertListEqual = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertMultiLineEqual = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertSequenceEqual  = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertRegexpMatches  = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertAlmostEquals   = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)
    #    assertItemsEqual     = property(fget=ForbidCalling,doc=ForbidCalling.__doc__)

    @RaiseError
    def assertEqual(self, clientAns, graderAns, suppressErr=False):
        return self.errorHandler.TestEqual(clientAns, graderAns, suppressErr=suppressErr)

    @RaiseError
    def assertEqualAmong(self, clientAns, graderAnsList, lastErrOnly=True, suppressErr=False):
        return self.errorHandler.TestEqualAmong(clientAns, graderAnsList, lastErrOnly=lastErrOnly,
                                                suppressErr=suppressErr)

    @RaiseError
    def assertAlmostEqual(self, clientAns, graderAns, places=5, suppressErr=False):
        self.errorHandler.SetAccuracy(places=places)
        return self.errorHandler.TestEqual(clientAns, graderAns, almost=True, suppressErr=suppressErr)

    @RaiseError
    def assertAlmostEqualAmong(self, clientAns, graderAnsList, places=5, lastErrOnly=True, suppressErr=False):
        self.errorHandler.SetAccuracy(places=places)
        return self.errorHandler.TestEqualAmong(clientAns, graderAnsList, almost=True, lastErrOnly=lastErrOnly,
                                                suppressErr=suppressErr)

    @RaiseError
    def assertGreater(self, clientAns, graderAns, suppressErr=False):
        return self.errorHandler.TestGreater(clientAns, graderAns, suppressErr=suppressErr)

    @RaiseError
    def assertGreaterEqual(self, clientAns, graderAns, suppressErr=False):
        return self.errorHandler.TestGreater(clientAns, graderAns, suppressErr=suppressErr, equal=True)

    @RaiseError
    def assertLess(self, clientAns, graderAns, suppressErr=False):
        return self.errorHandler.TestLess(clientAns, graderAns, suppressErr=suppressErr)

    @RaiseError
    def assertLessEqual(self, clientAns, graderAns, suppressErr=False):
        return self.errorHandler.TestLess(clientAns, graderAns, suppressErr=suppressErr, equal=True)

    @RaiseError
    def assertIsInstance(self, clientAns, typeObj, suppressErr=False):
        return self.errorHandler.TestIsInstance(clientAns, typeObj, suppressErr=suppressErr)

    @RaiseError
    def assertIsLambdaFunc(self, func, suppressErr=False, positive=True, funcName=None):
        oldSticky = self.stickyBuffer
        self.stickyBuffer = True
        funcNameStr = ''
        if funcName is not None: funcNameStr = funcName + ' '
        self.addPreErrorMsg(
            'The tested obj %son IsLambdaFunc has no __name__ attribute; it is not even a function!' % (funcNameStr))
        self.assertHasAttr(func, '__name__', suppressErr=True)
        self.errorHandler.errorBuffer.PopMessage()
        self.stickyBuffer = oldSticky
        return self.errorHandler.TestIsLambdaFunc(func, suppressErr=suppressErr, positive=positive, funcName=funcName)

    @RaiseError
    def assertVarsExist(self, varDict, varNameList, varTypeList=None, suppressErr=False):
        if varTypeList is not None and len(varNameList) != len(varTypeList):
            raise ValueError('assertVarsExist varNameList and varTypeList need to be of the same length.')
        return self.errorHandler.TestVarsExist(varDict, varNameList, varTypeList, suppressErr=suppressErr)

    @RaiseError
    def assertFuncReturnType(self, func, returnType, *args, **kwargs):
        return self.errorHandler.TestFuncReturnType(func, returnType, *args, **kwargs)

    @RaiseError
    def assertFuncReturnValues(self, func, returnValues, inputValues=None, firstErrOnly=True, suppressErr=False,
                               funcName=None):
        return self.errorHandler.TestFuncReturnValues(func, returnValues, inputValues=inputValues,
                                                      firstErrOnly=firstErrOnly, \
                                                      suppressErr=suppressErr, funcName=funcName)

    @RaiseError
    def assertHasAttr(self, obj, attrName, suppressErr=False):
        self.stickyBuffer = True
        self.assertIsInstance(attrName, str, suppressErr=suppressErr)
        attr = getattr(obj, attrName, None)
        isNone = attr is None
        if isNone and not suppressErr: self.addPreErrorMsg(
            '%s is not an attribute of your %s object.' % (attrName, str(obj.__class__)))
        return not isNone

    @RaiseError
    def assertTrue(self, boolStatement, suppressErr=False):
        return self.errorHandler.TestTrue(boolStatement, suppressErr=suppressErr)

    @RaiseError
    def assertFalse(self, boolStatement, suppressErr=False):
        return self.errorHandler.TestFalse(boolStatement, suppressErr=suppressErr)

    @RaiseError
    def assertCheck_Output(self, statement, result, almost=False, dType=str, **kwargs):
        try:
            ans = check_output(statement, shell=True)
        except CalledProcessError as e:
            self.addPostErrorMsg(str(e))
            items = statement.split()
            if len(items) > 1 and items[0] == 'python' and items[1].endswith('.py'):
                try:
                    py_compile.compile(items[1], doraise=True)
                except Exception as e:
                    self.addPostErrorMsg(str(e))
                self.addPostErrorMsg(
                    'Check your script \"%s\" for syntax, script input variables unpack error, etc.' % (items[1]))
                self.addPostErrorMsg(
                    'Suggestion: Try to run \"%s\" in the console to figure out the problem' % (statement))
            else:
                self.addPostErrorMsg('Running %s in a shell causes unidentified error.' % (statement))
            self.assertEqual(0, 1, suppressErr=True)
        try:
            dType(ans.strip())
        except:
            self.addPreErrorMsg('Your output \"%s\" from the command \"%s\" cannot be converted to the data type \"%s\"' % (
                ans.strip(), statement, str(dType)))
            return False
        self.addPreErrorMsg('In testing the output of the command \"%s\",' % (statement))
        return self.errorHandler.TestEqual(dType(ans.strip()), dType(result), almost=almost)

    def addPreErrorMsg(self, message, sticky=False):
        if sticky: self.stickyBuffer = True
        self.errorHandler.RegisterMessage(message)

    def addPostErrorMsg(self, message):
        self.errorHandler.RegisterPostMsg(message)

    def addDefaultMsg(self, varNameList):
        if isinstance(varNameList, str): varNameList = [varNameList]
        varStr = ', '.join(varNameList)
        self.addPreErrorMsg('In testing your variable(s) %s,' % (varStr))

    def pushPreMsg(self, message):  # pop automatically after the thrown exception
        if self.stickyBuffer: raise ValueError("pushPreMsg cannot be used while stickyBuffer bit is on")
        self.errorHandler.RegisterMessage(message)

    def pushPostMsg(self, message): # pop automatically after the thrown exception
        if self.stickyBuffer: raise ValueError("pushPostMsg cannot be used while stickyBuffer bit is on")
        self.errorHandler.RegisterMessage(message)

    def appdPreMsg(self, message):
        self.stickyBuffer = True
        self.errorHandler.RegisterMessage(message)

    def appdPostMsg(self, message):
        self.stickyBuffer = True
        self.errorHandler.RegisterMessage(message)


class ErrorHandler(object):
    supportedTypes = [bool, int, float, str, set, tuple, list, dict, np.ndarray, np.matrix, Series, DataFrame]
    wrongTypeMessage = 'Your answer is not of the correct data type: '
    numpyFloat = ['float', 'float128', 'float16', 'float32', 'float64']
    numpyInt = ['int', 'int0', 'int16', 'int32', 'int64', 'int8']

    def __init__(self, name='autograder'):
        self.name = name
        self.errorBuffer = ErrorBuffer()
        self.eps = 1e-7
        self.dfHeadCount = 5
        self.pdSeriesHeadCount = 4

        self.typeDict = dict(list(zip(map(lambda t: str(t), self.supportedTypes), self.supportedTypes)))
        self.operators = {}
        self.operators['!='] = lambda x, y: x != y
        self.operators['>'] = lambda x, y: x > y
        self.operators['<'] = lambda x, y: x < y
        self.operators['>='] = lambda x, y: x >= y
        self.operators['<='] = lambda x, y: x <= y

        self.opStr = {}
        self.opStr['!='] = 'not%s equal to'
        self.opStr['>'] = 'greater than'
        self.opStr['<'] = 'less than'
        self.opStr['>='] = 'greater than or equal to'
        self.opStr['<='] = 'less than or equal to'

    def SetAccuracy(self, places):
        places = abs(places)
        self.eps = 10 ** (-places)

    def RegisterMessage(self, message):
        self.errorBuffer.RegisterMessage(message)

    def RegisterPostMsg(self, message):
        self.errorBuffer.RegisterPostMsg(message)

    def Clear(self):
        self.errorBuffer.Clear()

    def TestIsInstance(self, clientAns, typeObj, suppressErr=False):
        error = None
        if not isinstance(typeObj, list):
            typeList = [typeObj]
        else:
            typeList = typeObj
        for x in typeList:
            if not isinstance(x, type): raise ValueError('%s is not a data type' % (str(x)))
            if not isinstance(clientAns, x):
                error = 'The type \"%s\" is not the expected type \"%s\"' % (str(type(clientAns)), str(x))
            if error and not suppressErr: self.errorBuffer.RegisterMessage(error)
        if error:
            return (False)
        else:
            return (True)

    def TestIsLambdaFunc(self, func, suppressErr=False, positive=True, funcName=None):
        error = None
        if funcName is None: funcName = func.__name__
        test = isinstance(func, types.LambdaType) and func.__name__ == '<lambda>'
        test = test if positive else not test
        notStr = ' not' if positive else ''
        if test:
            return (True)
        else:
            error = 'The function \"%s\" is%s a lambda function' % (funcName, notStr)
            if not suppressErr:  self.errorBuffer.RegisterMessage(error)
            return (False)

    def TestVarsExist(self, varDict, varNameList, varTypeList, suppressErr=False):
        error = [None for x in varNameList]
        if isinstance(varNameList, str):  varNameList = [varNameList]
        if varTypeList is not None and not isinstance(varTypeList, list): varTypeList = [varTypeList]
        for idx, element in enumerate(varNameList):
            if element not in varDict:
                typeStr = '' if (varTypeList is None or varTypeList[idx] is None) else ' of %s' % (
                    str(varTypeList[idx]))
                error[idx] = 'We expect the variable %s%s to be defined by you.' % (element, typeStr)
            elif varTypeList and varTypeList[idx] and not isinstance(varDict[element], varTypeList[idx]):
                error[idx] = 'The variable i\'%s\' you defined is not of the correct type \"%s\" we expect' % (
                    element, str(varTypeList[idx]))
            if error[idx] and not suppressErr:
                self.errorBuffer.RegisterMessage(error[idx])

        if any(error):
            return (False)
        else:
            return (True)

    def TestFuncReturnType(self, func, returnType, *args, **kwargs):
        funcName = kwargs.get('funcName')
        if funcName is None: funcName = func.__name__
        if type(func) not in [types.FunctionType, types.MethodType, types.BuiltinMethodType, types.BuiltinFunctionType,
                              type(np.sin), types.LambdaType]:
            self.errorBuffer.RegisterMessage('Your \"%s\" is not a function or method.' % (funcName))
            return (False)
        try:
            x = func(*args)
        except Exception as e:
            self.errorBuffer.RegisterMessage(
                'Your function \"%s\" cannot accept our argument \"%s\": ' % (funcName, str(args)) + str(e))
            self.errorBuffer.RegisterPostMsg(
                'Try to evaluate your function \"%s\" on some input (or our input) to debug.' % (funcName))
            return (False)
        if returnType is None: returnType = type(None)
        testAns = any([isinstance(x, t) for t in returnType]) if isinstance(returnType, list) else isinstance(x,
                                                                                                              returnType)
        if not testAns:
            typeStr = ' or '.join(map(lambda x: str(x), returnType)) if isinstance(returnType, list) else str(
                returnType)
            self.errorBuffer.RegisterMessage(
                'We expect your function \"%s\" to return type \"%s\",\nbut it returns \"%s\" instead.' % (
                    funcName, typeStr, type(x)))
            if x is None and returnType != type(None):
                self.errorBuffer.RegisterMessage(
                    'Did you forget to add a \"return\" statement in your function definition?')
            return (False)
        return (True)

    def TestFuncReturnValues(self, func, returnValues, inputValues=None, firstErrOnly=False, suppressErr=False,
                             funcName=None):
        if funcName is None: funcName = func.__name__
        if not isinstance(returnValues, list) and not isinstance(returnValues, tuple):
            returnValues = [returnValues]  # wrap it into a list if the user forgets to
            inputValues = [inputValues]
        if inputValues is not None:
            if len(returnValues) != len(inputValues):
                raise ValueError("returnValues and inputValues must be of the same length.")
            pairs = list(zip(inputValues, returnValues))
        else:
            pairs = list(zip([None for t in returnValues], returnValues))
        self.errorBuffer.RegisterMessage('In testing the return values of the function \"%s\",: ' % (funcName))
        self.errorBuffer.RegisterPostMsg('Your function \"%s\" fails to return the correct value.' % (funcName))
        ok = []
        for input, value in pairs:
            try:
                y = func(*input) if input is not None else func()
            except TypeError as e:
                lenInput = len(str(input))
                lenInput = 15 if lenInput > 15 else lenInput
                self.errorBuffer.RegisterMessage(
                    'Your function \"%s\" cannot accept our input \"%s\".' % (funcName, str(input)))
                self.errorBuffer.RegisterMessage('Due to TypeError: ' + str(e))
                self.errorBuffer.RegisterMessage('Please double-check your function \"%s\" definition.' % (funcName))
                self.errorBuffer.RegisterPostMsg(
                    'Suggestion: Evaluate the \"%s\" function with \"%s\" to debug' % (funcName, str(input)[0:lenInput]))
                return (False)
            except NameError as e:
                self.errorBuffer.RegisterMessage(
                    'Due to NameError: ' + str(e) + ', check the definition of \"%s\" to correct it.' % (funcName))
                return (False)
            except Exception as e:
                exceptionCall = e.__repr__()
                idx = exceptionCall.find('(')
                if idx > -1: exceptionCall = exceptionCall[
                                             :idx]  # extract the portion of error message we are interested in
                lenInput = len(input)
                self.errorBuffer.RegisterMessage('With input variable \"%s\",' % (str(input[0:lenInput])))
                self.errorBuffer.RegisterPostMsg(
                    'Suggestion: Evaluate the function \"%s\" with \"%s\" to debug.' % (funcName, str(input[0:lenInput])))
                self.errorBuffer.RegisterMessage(exceptionCall + ", " + str(e))
                return (False)
            stry = str(y) if len(str(y)) < 100 else str(y)[:100] + '...'  # keep the first 100 characters
            error = 'We test the return values of the input %s -> %s' % (str(input), stry)
            if not suppressErr and (
                        (not firstErrOnly) or (firstErrOnly and sum(ok) == len(ok))):  # up to now there is no error
                self.errorBuffer.RegisterMessage(error)
            if y is None and value is not None:
                self.errorBuffer.RegisterMessage(
                    'Make sure your function \"%s\" return is of the type \"%s\".' % (funcName, type(value)))
            ok.append(self.TestEqual(y, value, almost=True, suppressErr=suppressErr))
            if not suppressErr and (firstErrOnly and (sum(ok) == len(ok) or (sum(ok) < len(ok) - 1 and not ok[-1]))):
                # if we do not suppressErr and only keep the first error
                # either there is no error yet, or the error has occured, but not the latest one
                self.errorBuffer.PopMessage()

        if all(ok):
            return (True)
        else:
            return (False)

    def TestTrue(self, boolStatement, suppressErr=False):
        if boolStatement:
            return (True)
        else:
            if not suppressErr:
                self.errorBuffer.RegisterMessage('It evaluates to be false.')
                return (False)

    def TestFalse(self, boolStatement, suppressErr=False):
        if not boolStatement:
            return (True)
        else:
            if not suppressErr:
                self.errorBuffer.RegisterMessage('It evaluates to be true.')
                return (False)

    def TestEqual(self, clientAns, graderAns, almost=False, suppressErr=False, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        if operator is None: raise ValueError("\"%s\" is an invalid operator key." % (opKey))
        supportedTypes = self.supportedTypes if opKey == '!=' else [x for x in self.supportedTypes if
                                                                    x not in [set, dict]]
        flags = [isinstance(graderAns, x) for x in supportedTypes]
        if not any(flags):
            raise TypeError("ErrorHandler::TestEqual, with operator %s ---> this type %s is not supported." % (
                opKey, str(type(graderAns))))

        dataType = [str(x) for x in supportedTypes if isinstance(graderAns, x)][0]

        # if dataType == str(int):
        #    error = self.TestIntEqual(clientAns, graderAns, opKey)
        if dataType in str(bool):
            error = self.TestBoolEqual(bool(clientAns), graderAns)
        elif dataType in [str(float), str(int)]:
            error = self.TestFloatEqual(clientAns, graderAns, almost, opKey)
        elif dataType == str(str):
            error = self.TestStrEqual(clientAns, graderAns, opKey)
        elif dataType == str(set):
            error = self.TestSetEqual(clientAns, graderAns)
        elif dataType == str(list) or dataType == str(tuple):
            error = self.TestListEqual(clientAns, graderAns, opKey)
        elif dataType == str(dict):
            error = self.TestDictEqual(clientAns, graderAns)
        elif dataType == str(np.ndarray) or dataType == str(np.matrix):
            error = self.TestNumpyEqual(clientAns, graderAns, almost)
        elif dataType == str(Series):
            error = self.TestSeriesEqual(clientAns, graderAns, almost)
        elif dataType == str(DataFrame):
            error = self.TestDataFrameEqual(clientAns, graderAns, almost)
        else:
            raise ValueError("bad type")

        if error and not suppressErr: self.errorBuffer.RegisterMessage(error)

        if error:
            return (False)
        else:
            return (True)

    def TestEqualAmong(self, clientAns, graderAnsList, almost=False, lastErrOnly=True, suppressErr=False):
        '''
        If the clientAns matches with one of the elements in graderAnsList, return True, else return False
        the graderAnsList starts with the most desirable answer 
        '''
        if not isinstance(graderAnsList, list) and not isinstance(graderAnsList, tuple):
            raise ValueError("graderAnsList must be a Python list.")
        error = None
        X = [(x, isinstance(clientAns, type(x)), idx if lastErrOnly else -idx) for idx, x in enumerate(graderAnsList)]
        # when lastErrOnly, place the most desirable answer at the end after sorting
        Y = sorted(X, key=lambda x: (x[1], x[2]), reverse=True)
        Z = list(zip(*Y))
        if not any(Z[1]):
            error = 'Your answer type \"%s\" does not match with any of ours,' % (str(type(clientAns)))
            if not suppressErr: self.errorBuffer.RegiesterMessage(error)
            error = 'Our accepted answer types are \"%s\"' % (',\t'.join([str(type(x)) for x in graderAnsList]))
            if not suppressErr: self.errorBuffer.RegiesterMessage(error)
            return (False)
        else:
            for idx, element in enumerate(Z[0][:len(Z[1])]):
                ithStr = '%d-' % (idx + 1)
                if (idx + 1) % 10 == 1:
                    ithStr += 'st'
                elif (idx + 1) % 10 == 2:
                    ithStr += 'nd'
                elif (idx + 1) % 10 == 3:
                    ithStr += 'rd'
                else:
                    ithStr += 'th'
                if lastErrOnly and idx > 0:
                    self.errorBuffer.PopMessage()
                # the previous one must have failed (or we would have returned True aleady,
                # but we remove it for adding the next one
                elif not lastErrOnly and not suppressErr:
                    self.errorBuffer.RegisterMessage("For the %s correct answer: " % (ithStr))
                ok = self.TestEqual(clientAns, element, almost=almost, suppressErr=suppressErr)
                if ok: return (True)

            return (False)

    def TestBoolEqual(self, clientAns, graderAns, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % ('')
        if not isinstance(clientAns, int) and not isinstance(clientAns, float):
            error = '%s int' % (self.wrongTypeMessage)
        elif operator(clientAns, graderAns):
            error = 'Your answer \"%s\" is %s the %s\'s answer \"%s\".' % (str(clientAns), opStr, self.name, str(graderAns))

        return (error)

    def TestIntEqual(self, clientAns, graderAns, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % ('')
        if not isinstance(clientAns, int) and not isinstance(clientAns, float):
            error = '%s int' % (self.wrongTypeMessage)
        elif operator(clientAns, graderAns):
            if isinstance(clientAns, float):
                error = 'Your answer \"%f\" is %s the %s\'s answer \"%d\".' % (clientAns, opStr, self.name, graderAns)
            else:
                error = 'Your answer \"%d\" is %s the %s\'s answer \"%d\"' % (clientAns, opStr, self.name, graderAns)
        return (error)

    def TestFloatEqual(self, clientAns, graderAns, almost=False, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % (' approximately' if almost else '')
        if not isinstance(clientAns, float) and not isinstance(clientAns, int):
            error = '%s: float' % (self.wrongTypeMessage)
        elif (operator(clientAns, graderAns) and not almost) or (abs(clientAns - graderAns) > self.eps and almost):
            error = 'Your answer \"%f\" is %s the %s\'s answer \"%f\".' % (clientAns, opStr, self.name, graderAns)

        return (error)

    def TestStrEqual(self, clientAns, graderAns, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % ('')
        if not isinstance(clientAns, str):
            error = '%s str, it is of type \"%s\".' % (self.wrongTypeMessage, str(type(clientAns)))
        elif operator(clientAns, graderAns):
            error = 'Your answer \"%s\",\nis %s %s\'s answer \"%s\"' % (clientAns, \
                                                                        opStr, self.name, graderAns)

        return (error)

    def TestSetEqual(self, clientAns, graderAns):
        error = None
        if not isinstance(clientAns, set):
            error = '%s set, it is of type \"%s\".' % (self.wrongTypeMessage, str(type(clientAns)))
        elif len(clientAns) != len(graderAns):
            error = 'Your set is of cardinality \"%d\" while ours is of cardinality \"%d\". They are not equal.' % (
                len(clientAns), len(graderAns))
        elif clientAns != graderAns:
            if len(graderAns) < 25:
                error = 'Your set is: \"%s\"\n while ours is \"%s\".' % (str(clientAns), str(graderAns))
            else:
                error = 'Even though your set is of the same cardinality as ours, they are not identical.'

        return (error)

    def TestListEqual(self, clientAns, graderAns, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % ('')
        graderType = type(graderAns)
        if not isinstance(clientAns, list) and not isinstance(clientAns, tuple):
            graderAnsType = 'list' if isinstance(graderAns, list) else 'tuple'
            error = '%s %s, it is of type \"%s\".' % (self.wrongTypeMessage, graderAnsType, str(type(clientAns)))
        elif len(clientAns) != len(graderAns):
            error = 'The length of your list, %d, and the length of our list, %d, are different.' \
                    % (len(clientAns), len(graderAns))
            return (error)
        elif operator(graderType(clientAns), graderAns):
            clientAnsStr = str(clientAns)
            clientAnsStr = clientAnsStr[:100] + ' ...' if len(clientAnsStr) > 120 else clientAnsStr
            graderAnsStr = str(graderAns)
            graderAnsStr = graderAnsStr[:100] + ' ...' if len(graderAnsStr) > 120 else graderAnsStr
            error = 'Your answer:\n%s\nis %s ours:\n%s,\ndespite that they are of the same length.' % (
                clientAnsStr, opStr, graderAnsStr)
        return (error)

    def TestDictEqual(self, clientAns, graderAns):
        error = None
        if not isinstance(clientAns, dict):
            error = '%s dict, it is of type \"%s\".' % (self.wrongTypeMessage, str(type(clientAns)))
        elif len(clientAns.keys()) != len(graderAns.keys()):
            error = 'The length of your dictionary \"%d\" is not the same as ours (%d).' % (len(clientAns), len(graderAns))
        elif sorted(clientAns.keys()) != sorted(graderAns.keys()):
            x = str(clientAns.keys()) if len(str(clientAns.keys())) < 75 else str(clientAns.keys())[:70] + "..."
            y = str(graderAns.keys()) if len(str(graderAns.keys())) < 75 else str(graderAns.keys())[:70] + "..."
            error = 'The keys of your dictionary \"%s\" do not match with ours, \"%s\".' % (x, y)
        elif sorted(clientAns.items()) != sorted(graderAns.items()):
            x = str(sorted(clientAns.items()))
            y = str(sorted(graderAns.items()))
            x = x[:70] + ' ...' if len(x) > 75 else x
            y = y[:70] + ' ...' if len(y) > 75 else y
            error = 'Your (sorted) dictionary,\n\"%s\", does not coincide with ours,\n \"%s\".' % (x, y)

        return (error)

    def TestNumpyEqual(self, clientAns, graderAns, almost=False, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % (' approximately' if almost else '')
        if not isinstance(clientAns, np.ndarray) and not isinstance(clientAns, np.matrix):
            error = '%s np.array, it is of type \"%s\".' % (self.wrongTypeMessage, str(type(clientAns)))
        elif clientAns.shape != graderAns.shape:
            error = 'The shape of your numpy array \"%s\" is not equal to ours\' shape %s' \
                    % (str(clientAns.shape), str(graderAns.shape))

        elif (operator(clientAns, graderAns).any() and not almost) or (
                    almost and np.abs(clientAns - graderAns).max() > self.eps):
            endStr = ''
            if len(graderAns.shape) == 1:
                x = clientAns[:50] if clientAns.shape[0] >= 50 else clientAns
                y = graderAns[:50] if graderAns.shape[0] >= 50 else graderAns
                endStr = '...'
            elif len(graderAns.shape) == 2:
                x = clientAns[:self.dfHeadCount, :]
                y = graderAns[:self.dfHeadCount, :]
                endStr = '\n.\n.\n.\n'
            else:
                x = clientAns
                y = graderAns
            clientAnsStr = str(x) + endStr
            graderAnsStr = str(y) + endStr
            error = 'Your answer, with top %d lines:\n%s\nis %s ours (top %d lines shown below):\n%s' \
                    % (self.dfHeadCount, clientAnsStr, opStr, self.dfHeadCount, graderAnsStr)

        return (error)

    def TestSeriesEqual(self, clientAns, graderAns, almost=False, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % (' approximately' if almost else '')
        if not isinstance(clientAns, Series):
            error = '%s pd.Series' % (self.wrongTypeMessage)
        elif clientAns.shape != graderAns.shape:
            error = 'The shape of your pandas Series \"%s\" is not equal to ours\' shape %s' \
                    % (str(clientAns.shape), str(graderAns.shape))
        elif (clientAns.index != graderAns.index).any():
            clientIdxStr = str(clientAns.index)[:50] + " ..." if len(str(clientAns.index)) >= 50 else str(
                clientAns.index)
            graderIdxStr = str(graderAns.index)[:50] + " ..." if len(str(graderAns.index)) >= 50 else str(
                graderAns.index)
            error = ' Your pd.Series index,\n%s,\n does not match with ours,\n%s' % (clientIdxStr, graderIdxStr)
        elif (operator(clientAns, graderAns).any() and not almost) or (
                    almost and (clientAns - graderAns).abs().max() > self.eps):
            appro = 'approximately' if almost else ''
            clientAnsStr = str(clientAns.head(self.pdSeriesHeadCount))
            graderAnsStr = str(graderAns.head(self.pdSeriesHeadCount))
            error = ' Your pd.Series answer (top %d lines shown below):\n%s\n is not %s equal to ours (top %d lines shown below):\
               \n%s' % (self.pdSeriesHeadCount, clientAnsStr, appro, self.pdSeriesHeadCount, graderAnsStr)

        return (error)

    def TestDataFrameEqual(self, clientAns, graderAns, almost=False, opKey='!='):
        error = None
        operator = self.operators.get(opKey)
        opStr = self.opStr.get(opKey) % (' approximately' if almost else '')
        if not isinstance(clientAns, DataFrame):
            error = '%s pd.DataFrame, it is of type \"%s\".' % (self.wrongTypeMessage, str(type(clientAns)))
        elif clientAns.shape != graderAns.shape:
            error = 'The shape of your pandas DataFrame \"%s\" is not equal to ours\' shape %s.' \
                    % (str(clientAns.shape), str(graderAns.shape))
        elif (clientAns.index != graderAns.index).any():
            clientIdxStr = str(clientAns.index)[:50] + " ..." if len(str(clientAns.index)) >= 50 else str(
                clientAns.index)
            graderIdxStr = str(graderAns.index)[:50] + " ..." if len(str(graderAns.index)) >= 50 else str(
                graderAns.index)
            error = ' Your pd.DataFrame index,\n%s,\ndoes not match with ours,\n%s.' % (clientIdxStr, graderIdxStr)
        elif (clientAns.columns != graderAns.columns).any():
            clientColStr = str(clientAns.columns)[:50] + " ..." if len(str(clientAns.columns)) >= 50 else str(
                clientAns.columns)
            graderColStr = str(graderAns.columns)[:50] + " ..." if len(str(graderAns.columns)) >= 50 else str(
                graderAns.columns)
            error = ' Your pd.DataFrame columns,\n%s,\ndoes not match with ours,\n%s.' % (clientColStr, graderColStr)

        elif (operator(clientAns, graderAns).any().any() and not almost) or (
                    almost and (clientAns - graderAns).abs().values.max() > self.eps):
            endStr = '' if graderAns.shape[0] <= self.dfHeadCount else '\n.\n.\n.\n'
            x = str(clientAns.head(self.dfHeadCount)) + endStr
            y = str(graderAns.head(self.dfHeadCount)) + endStr
            error = 'Your answer, with the top %d lines:\n%s\n, is %s ours (top %d lines shown below):\n%s' \
                    % (self.dfHeadCount, x, opStr, self.dfHeadCount, y)

        return (error)

    def TestGreater(self, clientAns, graderAns, suppressErr=False, equal=False):
        opKey = '<=' if not equal else '<'
        return (self.TestEqual(clientAns, graderAns, suppressErr=suppressErr, opKey=opKey))

    def TestLess(self, clientAns, graderAns, suppressErr=False, equal=False):
        opKey = '>=' if not equal else '>'
        return (self.TestEqual(clientAns, graderAns, suppressErr=suppressErr, opKey=opKey))


class ErrorBuffer(object):

    def __init__(self):
        self.messageBuffer = []
        self.postMsgBuffer = []

    def Clear(self):
        self.messageBuffer = []
        self.postMsgBuffer = []

    def PopMessage(self):
        return self.messageBuffer.pop()

    def PopPostMsg(self):
        return self.postMsgBuffer.pop()

    def RegisterMessage(self, message):
        self.messageBuffer.append(message)

    def RegisterPostMsg(self, message):
        self.postMsgBuffer.append(message)

    def OutputErrors(self, clear=True):
        message = '\n' + '\n'.join(self.messageBuffer + self.postMsgBuffer)
        if clear:  self.Clear()
        return (message)
