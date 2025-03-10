# %% [markdown]
# # parser
# * 输入: 一个标记列表
# * 输出: 一个 AST 节点列表

# %%
from abc import ABC, abstractmethod

class AstNode(ABC):
    
    @abstractmethod
    def __init__(self):
        self.quantative = False
        self.min_repeat = 1
        self.max_repeat = 1
        self.greedy = False
        self.start = False
        self.end = False
    
    def getState(self):
        pattern = ""
        if self.quantative:
            pattern += f" min: {self.min_repeat}, max: {self.max_repeat}, greedy: {self.greedy}"
        if self.start:
            pattern += " start"
        if self.end:
            pattern += " end"
        return pattern
    
    @abstractmethod
    def print(self, depth=0):
        pass

class OrAstNode(AstNode):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
    def print(self, depth=0):
        print(" " * depth + "OR", self.getState())
        self.left.print(depth + 1)
        self.right.print(depth + 1)

class SeqAstNode(AstNode):
    def __init__(self, children: list[AstNode] = None):
        super().__init__()
        if children is None:
            children = []
        self.children = children
    def print(self, depth=0):
        print(" " * depth + "SEQ", self.getState())
        for child in self.children:
            child.print(depth + 1)
        

class LiteralGroupAstNode(AstNode):
    def __init__(self, chars: list[int] = None):
        super().__init__()
        if chars is None:
            chars = []
        self.chars = chars
    
    def print(self, depth=0):
        print(" " * depth + "LITERAL_GROUP", self.getState())

class LiteralCharacterAstNode(LiteralGroupAstNode):
    def __init__(self, char: int):
        super().__init__([char])
        
    def print(self, depth=0):
        print(" " * depth + "LITERAL", chr(self.chars[0]), self.getState())

class LiteralSpecialCharacterAstNode(LiteralGroupAstNode):
    def __init__(self, reversed:bool, isWord=False, isDigit=False, isSpace=False):
        assert isWord or isDigit or isSpace, "at least one of isWord, isDigit, isSpace must be True"
        assert isWord + isDigit + isSpace == 1, "only one of isWord, isDigit, isSpace can be True"
        chars = []
        if isWord:
            chars = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_']
            self.description = 'word'
        elif isDigit:
            chars = [ord(c) for c in '0123456789']
            self.description = 'digit'
        elif isSpace:
            chars = [ord(c) for c in '  \n\r\f\v']
            self.description = 'space'
        if reversed:
            self.description += ' reversed'
            newchars = []
            for i in range(256):
                if i not in chars:
                    newchars.append(i)
            chars = newchars
        super().__init__(chars)
    def print(self, depth=0):
        print(" " * depth + "LITERAL_SPECIAL_CHARACTER", self.getState())

class SquareBracketAstNode(AstNode):
    def __init__(self, children: list[AstNode] = [], reversed = False):
        super().__init__()
        self.children = children
        self.reversed = reversed
    def print(self, depth=0):
        print(" " * depth + "SQUARE_BRACKET", [child.chars[0] for child in self.children], self.getState())
    
    def getState(self):
        return super().getState() + f" reversed: {self.reversed}"
