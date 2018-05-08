# import time
#
# def time_it(fn):
#     print ('time_it is executed')
#     def new_fn(*args):
#         start = time.time()
#         result = fn(*args)
#         end = time.time()
#         duration = end - start
#         print('%s seconds are consumed in executing function:%s%r'\
#               %(duration, fn.__name__, args))
#         return result
#
#     return new_fn
#
# @time_it
# def acc1(start, end):
#     s = 0
#     for i in range(start, end):
#         s += i
#     return s
#
#
# def acc2(start, end):
#     s = 0
#     for i in range(start, end):
#         s += i
#     return s
#
# print(acc1)
# print(acc2)
#
# if __name__ == '__main__':
#     acc1(10, 1000000)


def log(func):
    def wrapper(*args, **kv):
        print('call %s():'%func.__name__)
        return func(*args, **kv)
    return wrapper

@log
def now():
    print('2015-03-25')

now()

def log2(text):
    def decorator(func):
        def warpper(*args, **kv):
            print('%s %s'% (text, func.__name__))
            return func(*args, **kv)
        return warpper
    return decorator

@log2('call')
def now2():
    print('2018-05-05')

now2()

# def log3(text):
#     def decorator(func):
#         def warpper(*args, **kv):
#             print('%s %s'% (text, func.__name__))
#             return func(*args, **kv)
#         return warpper
#     return decorator
#
# @log3
# def now3():
#     print('2018-05-05')
#
# new_fn = log('run')(now3)
# new_fn('kejian')

import functools
def log3(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kv):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kv)
        return wrapper
    return decorator

@log3('kejian')
def fn(str):
    print('hello world ' + str)

def log4(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kv):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kv)
        return wrapper
    return decorator


fn('kejian')
print('--------------------------------')
newfn = log4('andy')(fn)
newfn('andy')
print('--------------------------------')
import collections
point = collections.namedtuple('kejian', ['age', 'height'])
p = point(25, 180)
print(p[1])
name1, age1 = p
print(name1)

print(p.age)






