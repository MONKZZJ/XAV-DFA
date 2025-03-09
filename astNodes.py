# %% [markdown]
# # parser
# * 输入: 一个标记列表
# * 输出: 一个 AST 节点列表

# %%
from abc import ABC, abstractmethod

class AstNode(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def get_class_name(self):
        return self.__class__.__name__

class QuantativeAstNode(AstNode):
    def __init__(self):
        self.lazy = False

class OrAstNode(AstNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
class SeqAstNode(AstNode):
    def __init__(self, left, right):
        not_group = False
        self.children = []
        if isinstance(left, SeqAstNode):
            self.children.extend(left.children)
        else:
            self.children.append(left)
        if isinstance(right, SeqAstNode):
            self.children.extend(right.children)
        else:
            self.children.append(right)
class StarAstNode(QuantativeAstNode):
    def __init__(self, left):
        super().__init__()
        self.left = left
class PlusAstNode(QuantativeAstNode):
    def __init__(self, left):
        super().__init__()
        self.left = left
class QuestionMarkAstNode(AstNode):
    def __init__(self, left):
        self.left = left
'''
class STARTAstNode(PlusAstNode, QuestionMarkAstNode, AstNode):
    def __init__(self, right):
        super().__init__(False)
        self.right = right
'''

class STARTAstNode(AstNode):
    def __init__(self, right, parent_node=None):
        self.right = right
        self.parent_node = parent_node  # 组合其他节点

    def some_method(self):
        if self.parent_node:
            # 调用父节点的方法
            self.parent_node.some_method()

# 示例：创建 STARTAstNode 并组合其他节点
# start_node = STARTAstNode(right=some_node, parent_node=PlusAstNode(False))

class LiteralCharacterAstNode(AstNode):
    def __init__(self, char):
        self.char = char
class SquareBracketAstNode(AstNode):
    # clas: set #of strs and pairs
    # for example: [a-z] -> {'a', 'b', 'c', ..., 'z'}
    # [a-z0-9] -> {'a', 'b', 'c', ..., 'z', '0', '1', ..., '9'
    # [a-Z012] -> {'a', 'b', 'c', ..., 'Z', '0', '1', '2'}
    def __init__(self, clas):
        self.clas = clas
        self.negated = False
        self.ranges = []

    def add_range(self, start, end):
        self.ranges.append((start, end))

class QuantifierAstNode(QuantativeAstNode):
    def __init__(self, left, min, max=None):
        super().__init__()
        self.left = left
        self.min = min
        self.max = max
class AssertAstNode(AstNode):
    def __init__(self, child, is_positive=True):
        self.child = child
        self.is_positive = is_positive
class BackReferenceAstNode(AstNode):
    def __init__(self, index):
        self.index = index
class NonCapturingGroupAstNode(AstNode):
    def __init__(self, child):
        self.child = child


def print_ast(node, indent=0):
    """
    打印AST节点

    参数:
    node (AstNode): AST节点
    indent (int): 缩进级别
    """
    if isinstance(node, OrAstNode):
        print(' ' * indent + 'OR')
        print_ast(node.left, indent + 2)
        print_ast(node.right, indent + 2)
    elif isinstance(node, STARTAstNode):
        print(' ' * indent + 'START_WITH')
        print_ast(node.right, indent + 2)
    elif isinstance(node, SeqAstNode):
        print(' ' * indent + 'SEQ')
        for child in node.children:
            print_ast(child, indent + 2)
    elif isinstance(node, StarAstNode):
        print(' ' * indent + ('LAZY' if node.lazy else '')  + 'STAR')
        print_ast(node.left, indent + 2)
    elif isinstance(node, PlusAstNode):
        print(' ' * indent + ('LAZY' if node.lazy else '') +  'PLUS')
        print_ast(node.left, indent + 2)
    elif isinstance(node, QuestionMarkAstNode):
        print(' ' * indent + 'QUESTION_MARK')
        print_ast(node.left, indent + 2)
    elif isinstance(node, LiteralCharacterAstNode):
        print(' ' * indent + 'LITERAL: ' + str(node.char))
    elif isinstance(node, SquareBracketAstNode):
        print(' ' * indent + ('NEG-' if node.negated else '') + 'SQUARE_BRACKET')
        if node.clas:
            for char in node.clas:
                if isinstance(char, tuple):
                    print(' ' * (indent + 2) + 'RANGE: {}-{}'.format(char[0], char[1]))
                else:
                    print(' ' * (indent + 2) + 'CHARACTER: {}'.format(char))
        if node.ranges:
            for start, end in node.ranges:
                print(' ' * (indent + 2) + 'RANGE: {}-{}'.format(chr(start), chr(end)))
    elif isinstance(node, AssertAstNode):
        print(' ' * indent + ('POSITIVE' if node.is_positive else 'NEGATIVE') + 'ASSERT')
        print_ast(node.child, indent + 2)
    elif isinstance(node, BackReferenceAstNode):
        print(' ' * indent + 'BACKREFERENCE: {}'.format(node.index))
    elif isinstance(node, QuantifierAstNode):
        print(' ' * indent + 'QUANTIFIER: {}-{}'.format(node.min, node.max))
        print_ast(node.left, indent + 2)
    elif isinstance(node, NonCapturingGroupAstNode):
        print(' ' * indent + 'NON_CAPTURING_GROUP')
        print_ast(node.child, indent + 2)
    else:
        raise ValueError('Invalid AST node type')

