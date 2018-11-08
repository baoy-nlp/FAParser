#coding:utf-8
"""
Program: 栈和队列的实现
Description: 
Author: Flyaway - flyaway1217@gmail.com
Date: 2014-03-30 09:30:17
# Last modified: 2014-12-19 10:56:15
Python release: 3.3.1
"""

class Stack:
    def __init__(self):
        self.stack = []

    def isEmpty(self):
        return len(self.stack) <= 0

    def top(self):
        if self.isEmpty() == True:
            raise Exception('StackIsEmpty')
        else:
            x = self.stack[-1]
            return x

    def push(self,data):
        self.stack.append(data)

    def pop(self):
        if self.isEmpty() == True:
            raise Exception('StackIsEmpty')
        else:
            x = self.stack[-1]
            self.stack = self.stack[:-1]
            return x

    def show(self):
        '''
        打印出栈中的数据
        '''
        print(self.stack)

    def clear(self):
        self.stack=[]

    def __len__(self):
        return len(self.stack)

class Queue:
    def __init__(self):
        self.queue = []
    def __len__(self):
        return len(self.queue)
    def isEmpty(self):
        if len(self.queue) <= 0:
            return True
        else:
            return False 
    def enqueue(self,data):
        self.queue.append(data)
    def dequeue(self):
        if self.isEmpty() == True:
            raise Exception('QueueIsEmpty')
        else:
            x = self.queue[0]
            self.queue = self.queue[1:]
            return x
    def show(self):
        '''
        打印出队列中的信息
        '''
        print(self.queue)
    def clear(self):
        self.queue=[]


