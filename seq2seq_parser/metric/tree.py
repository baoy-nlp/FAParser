#coding:utf-8
"""

Program: 读取树结构的类，返回一个句法树的树结构

Description: 根据文本中的数据，在内存中生成一个树结构，用两个栈来完成。
数据格式举例:(IP (NP (NP (NR 上海) (NR 浦东)) (NP (NN 开发) (CC 与) (NN 法制) (NN 建设))) (VP (VV 同步)))

Author: Flyaway - flyaway1217@gmail.com

Date: 2014-05-23 20:01:43

# Last modified: 2015-01-26 10:04:45

"""
from Stack_Queue import Stack
from Stack_Queue import Queue


class ParseError(Exception):
    """The exception class of Parser

    Attributes:
        errormessage: the information of exception
    """
    def __init__(self,errormessage):
        self.errormessage = "Parsing Error: "
        self.errormessage += errormessage

class Node:
    '''The definition of node class of the Tree class

    Attributes:
        data: a string. if current node is a leaf , the data is word;if current node is not a leaf, the data is the tag
        children: a list of Node. if current node is a leaf, children == None

    '''
    def __init__(self,data,children=None):
        self.data = data.strip()
        if children == None:
            self.children =  []
        else:
            self.children = children

    def getData(self):
        '''Return the data of the node
        if the node is not a leaf, it returns the syntax tag or POS tag;
        if the node is a leaf, it returns the text of the word
        '''
        return self.data[:]

    def getChildren(self):
        '''
        return a list which contains all the child nodes
        '''
        return self.children[:]

    def isLeaf(self):
        return len(self.children) == 0

    def __repr__(self):
        s = 'Node:' + self.data
        return s

    def __str__(self):
        s = 'Node:' + self.data
        return s

class Tree:
    '''The structure of a tree

        Attributes:
            root: the root of the tree
            sentence: the plain sentence of the tree
    '''
    def __init__(self,sentence):
        """ Construct a tree from a string or another Tree instance

        Args:
            sentence: a string of the sentence in the format of brackets.
        """
        if type(sentence) == str:
            mysentence = sentence.replace(' (','(')
            mysentence = mysentence.replace(' )',')')
            mysentence = mysentence.replace('(',' ( ')
            mysentence = mysentence.replace(')',' ) ')

            self.root = Tree.MakeTree(mysentence)
            self.sentence = self.__getSentence(self.root)
        elif type(sentence) == Tree:
            self.root = sentence.root
            self.sentence = sentence.sentence
        elif type(sentence) == Node:
            self.root = sentence
            self.sentence = self.__getSentence(self.root)
        else:
            raise Exception("sentence error!")


    def __getSentence(self,node):
        """Get the plain sentence

        Getting the sentence using recursion

        Args:
            node: a tree node.

        Returns:
            a list of words in the sentence
            for example:
                ["上海","浦东","开发","与","法制","建设","同步"]
        """
        if node.isLeaf():
            return [node.getData()]
        else:
            partial = []
            for child in node.children:
                partial += self.__getSentence(child)
            return partial


    def MakeTree(cls,arg):
        """The core process of constructing a parser tree.

        Args:
            arg: a string of the sentence in the format of brackets.
        Returns:
            None
        """
        if len(arg.strip()) == 0:
            raise ParseError("Empty String")
        sentence = arg.split()
        stack = Stack()
        flag = False
        for i in range(len(sentence)-1):
            if sentence[i] == '(' and Tree.isTerminal(sentence[i+1]):
                stack.push(sentence[i])
                flag = True
            elif sentence[i] == '(' and sentence[i+1] == ')':
                if flag == False:
                    raise ParseError('No element in bracket')
                else:
                    node = Node(sentence[i])
                    stack.push(node)
                    flag = False
            elif sentence[i] == '(' and sentence[i+1] == '(':
                stack.push(sentence[i])
                node = Node('root')
                stack.push(node)
            elif sentence[i] == ')' and Tree.isTerminal(sentence[i+1]):
                raise ParseError('The format is wrong!')
            elif sentence[i] == ')' and sentence[i+1] == ')':
                if flag == False:
                    Tree.StackOperation(stack)
                else:
                    node = Node(sentence[i])
                    stack.push(node)
                    flag = False
            elif sentence[i] == ')' and sentence[i+1] == '(':
                if flag == False:
                    Tree.StackOperation(stack)
                else:
                    stack.push(Node(' '))
                    Tree.StackOperation(stack)
                    #raise Exception('Only one element in bracket')
            elif Tree.isTerminal(sentence[i]) and (sentence[i+1] == '(' or sentence[i+1] == ')'):
                node = Node(sentence[i])
                stack.push(node)
            elif Tree.isTerminal(sentence[i]) == True and Tree.isTerminal(sentence[i+1]) == True:
                if flag == False:
                    raise ParseError('There are more than 2 elements in bracket.')
                else:
                    node = Node(sentence[i])
                    stack.push(node)
                    flag = False

        Tree.StackOperation(stack)

        if len(stack) != 1:
            nodes = []
            while stack.isEmpty() == False:
                if isinstance(stack.top(),Node)==True:
                    nodes.append(stack.pop())
                else:
                    stack.pop()
            if len([node for node in nodes if node.getData()=='root']) > 1:
                raise ParseError("There more than 1 root node")
            else:
                raise ParseError("Unmathed bracket")
        return stack.pop()

    def StackOperation(cls,stack):
        """The operation on the stack

        Pop items and insert a new item

        Args:
            stack: a stack of nodes

        Returns:
            None
        """
        if len(stack) == 0:
            return
        children = []
        while stack.isEmpty()== False and stack.top() != '(' :
            children.append(stack.pop())
        parent = children.pop(-1)
        children.reverse()
        parent.children = children
        if len(stack) == 0:
            raise ParseError("Unmathed bracket")
        stack.pop()
        stack.push(parent)



    def isTerminal(cls,token):
        return not( (token == '(') or (token == ')') )


    MakeTree = classmethod(MakeTree)
    StackOperation= classmethod(StackOperation)
    isTerminal= classmethod(isTerminal)

    def to_bracket(self,node):
        # The pos tag node
        if len(node.children) == 1 and node.children[0].isLeaf() == True:
            pos_tag = node.getData()
            text = node.children[0].getData()
            s = "(" + pos_tag + " " + text+")"
            return s
        # The label tag node
        else:
            bracket_string = []
            for child in node.children:
                bracket_string.append(self.to_bracket(child))
            bracket_string = ' '.join(bracket_string)
            pos_tag = node.getData()
            if pos_tag == "root":
                bracket_string = "( " + bracket_string + " )"
            else:
                bracket_string = "(" + pos_tag + " " + bracket_string + ")"
            return bracket_string




    def __str__(self):
        return self.to_bracket(self.root)

    def __repr__(self):
        #return ' '.join(self.sentence)
        return self.to_bracket(self.root)

    def getRoot(self):
        return self.root

if __name__ == "__main__":
    s = "(  ( IP (VP  这是) (NP 测试) )  )"
    try:
        print(s)
        t = Tree(s)
        print(t)
    except ParseError as e:
        print(e.errormessage)

    #t.show()
    #test_path = "Processed_CTB/Bracket.txt"
    #with open(test_path) as f:
    #    for line in f:
    #        try: 
    #            t = Tree(line)
    #        except:
    #            print(line)
