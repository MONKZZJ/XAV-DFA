# %% [markdown]
# # lexer
import re

# * 输入: 一个字符串
# * 输出: 一个标记列表

in_PAREN = 0
in_sq_bracket = 0
in_BR = False

ast = None

# %%
from enum import Enum, auto


class TokenType(Enum):
    OR = auto()
    STAR = auto()
    PLUS = auto()
    QUESTION_MARK = auto()
    OPEN_PAREN = auto()
    CLOSED_PAREN = auto()
    OPEN_SQUARE_BRACKET = auto()
    CLOSED_SQUARE_BRACKET = auto()
    OPEN_BRACE = auto()
    CLOSED_BRACE = auto()
    DASH = auto()
    COMMA = auto()
    LITERAL = auto()
    WORD = auto()
    DIGIT = auto()
    SPACE = auto()
    NOTWORD = auto()
    NOTDIGIT = auto()
    NOTSPACE = auto()

    NOT = auto()
    TURE_ASSERT = auto()
    FALSE_ASSERT = auto()
    START = auto()
    BACK_REFERENCE = auto()
    NOT_GROUP = auto()
    DIGITS = auto()
    ANY = auto()


class Token:
    ttype: TokenType
    content: str

    def __init__(self, ttype, content: int = -1):
        self.ttype = ttype
        self.content = content

    def __repr__(self) -> str:
        return f"Token({self.ttype=}, {self.content=})"


def getNextToken(string: str, tokenStream: list[Token]) -> tuple[list[Token], int]:
    global in_PAREN, in_sq_bracket, in_BR
    token = string[0]
    nextoffset = 1
    returnTokens = []
    if token == '|':
        if in_sq_bracket:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        else:
            returnTokens.append(Token(TokenType.OR, -1))
    # elif token == '.':
    #     returnTokens.append(Token(TokenType.ANY, -1))
    elif token == '*':
        if in_sq_bracket:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        else:
            returnTokens.append(Token(TokenType.STAR, -1))
    elif token == '+':
        if in_sq_bracket:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        else:
            returnTokens.append(Token(TokenType.PLUS, -1))
    elif token == '?':
        if in_sq_bracket:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        else:
            returnTokens.append(Token(TokenType.QUESTION_MARK, -1))
    elif token == '(':
        returnTokens.append(Token(TokenType.OPEN_PAREN, -1))
        if string[1] == '?':
            if string[2] == '=':
                returnTokens.append(Token(TokenType.TURE_ASSERT, -1))
                nextoffset = 3
            elif string[2] == '!':
                returnTokens.append(Token(TokenType.FALSE_ASSERT, -1))
                nextoffset = 3
            elif string[2] == ":":
                returnTokens.append(Token(TokenType.NOT_GROUP, -1))
                nextoffset = 3
            else:
                raise ValueError("Invalid assertion,expected '=' or '!'")
    elif token == ')':
        returnTokens.append(Token(TokenType.CLOSED_PAREN, -1))
    elif token == '[':
        in_PAREN += 1
        in_sq_bracket += 1
        if string[1] == '^':
            returnTokens.append(Token(TokenType.OPEN_SQUARE_BRACKET, -1))
            returnTokens.append(Token(TokenType.NOT, -1))
            nextoffset = 2
        else:
            returnTokens.append(Token(TokenType.OPEN_SQUARE_BRACKET, -1))
    elif token == ']':
        in_PAREN -= 1
        in_sq_bracket -= 1
        returnTokens.append(Token(TokenType.CLOSED_SQUARE_BRACKET, -1))
    elif token == '-':
        # print(in_PAREN)
        if in_PAREN == 0:
            returnTokens.append(Token(TokenType.LITERAL, ord('-')))
        else:
            returnTokens.append(Token(TokenType.DASH, -1))
    elif token == '^':
        returnTokens.append(Token(TokenType.START, -1))
    elif token == '{':
        if len(string) == 1 or not string[1].isdigit():  # 直接判断 '{' 是否是最后一个字符或者后面不是数字
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        elif string[1].isdigit():
            in_BR = True
            in_PAREN += 1
            num_str = ''
            nextoffset = 1
            while nextoffset < len(string) and string[nextoffset].isdigit():
                num_str += string[nextoffset]
                nextoffset += 1
            min_val = int(num_str)
            max_val = 0
            if string[nextoffset] == ',':
                nextoffset += 1
                num_str = ''
                while nextoffset < len(string) and string[nextoffset].isdigit():
                    num_str += string[nextoffset]
                    nextoffset += 1
                if num_str:
                    max_val = int(num_str)
                else:
                    max_val = float('inf')
            if string[nextoffset] == '}':
                in_BR = False
                in_PAREN -= 1
                nextoffset += 1
                returnTokens.append(Token(TokenType.OPEN_BRACE, min_val))
                returnTokens.append(Token(TokenType.CLOSED_BRACE, max_val))
            else:
                raise ValueError("Invalid quantifier")
        else:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
    # elif token == '}':
    #     in_PAREN -= 1
    #     returnTokens.append(Token(TokenType.CLOSED_BRACE, -1))
    elif token == ',':
        if in_PAREN:
            returnTokens.append(Token(TokenType.COMMA, -1))
    elif token.isdigit():
        # print('DIG2LITE')
        returnTokens.append(Token(TokenType.LITERAL, ord(token)))
        # else:
        #    print('DIG')
        #    num_str = token
        #    while nextoffset < len(string) and string[nextoffset].isdigit():
        #        num_str += string[nextoffset]
        #        nextoffset += 1
        #    returnTokens.append(Token(TokenType.DIGITS, int(num_str)))
    elif token == '\\':
        char = string[1]
        nextoffset = 2
        if char.isdigit():
            returnTokens.append(Token(TokenType.BACK_REFERENCE, int(char)))
        elif char in ["+", "*", "?", "^", "$", "\\", ".", "-", "[", "]", "{", "}", "(", ")", "|", "/"]:
            #  get ascii code
            returnTokens.append(Token(TokenType.LITERAL, ord(char)))
        elif char == "0":
            returnTokens.append(Token(TokenType.LITERAL, int(string[2:4], base=8)))
            nextoffset = 4
        elif char == "x":
            returnTokens.append(Token(TokenType.LITERAL, int(string[2:4], base=16)))
            nextoffset = 4
        elif char == "u":
            if string[2] == "{":
                end_index = string.find("}")
                assert end_index != -1, "Invalid unicode escape sequence"
                assert (end_index - 3) % 2 == 0, "Invalid unicode escape sequence"
                for i in range(3, end_index, 2):
                    returnTokens.append(Token(TokenType.LITERAL, int(string[i:i + 2], base=16)))
                nextoffset = end_index + 1
            else:
                returnTokens.append(Token(TokenType.LITERAL, int(string[2:4], base=16)))
                returnTokens.append(Token(TokenType.LITERAL, int(string[4:6], base=16)))
                nextoffset = 6
        elif char == "c":
            assert not (ord("A") <= ord(string[2]) <= ord("Z") or ord("a") <= ord(string[2]) <= ord(
                "z")), "Invalid control character"
            if ord(string[2]) < "a":
                returnTokens.append(Token(TokenType.LITERAL, ord(string[2]) - ord("A") + 1))
            else:
                returnTokens.append(Token(TokenType.LITERAL, ord(string[2]) - ord("a") + 1))
        elif char == "t":
            returnTokens.append(Token(TokenType.LITERAL, ord("\t")))
        elif char == "n":
            returnTokens.append(Token(TokenType.LITERAL, ord("\n")))
        elif char == "v":
            returnTokens.append(Token(TokenType.LITERAL, ord("\v")))
        elif char == "r":
            returnTokens.append(Token(TokenType.LITERAL, ord("\r")))
        elif char == "f":
            returnTokens.append(Token(TokenType.LITERAL, ord("\f")))
        elif char == "0":
            returnTokens.append(Token(TokenType.LITERAL, ord("\0")))
        elif char == "w":
            returnTokens.append(Token(TokenType.WORD))
        elif char == "d":
            returnTokens.append(Token(TokenType.DIGIT))
        elif char == "s":
            returnTokens.append(Token(TokenType.SPACE))
        elif char == "W":
            returnTokens.append(Token(TokenType.NOTWORD))
        elif char == "D":
            returnTokens.append(Token(TokenType.NOTDIGIT))
        elif char == "S":
            returnTokens.append(Token(TokenType.NOTSPACE))
        else:
            returnTokens.append(Token(TokenType.LITERAL, ord(char)))
        # else:
        #     raise ValueError(f"here is a {char} which is Invalid escape sequence")

    else:
        returnTokens.append(Token(TokenType.LITERAL, ord(token)))
    return returnTokens, nextoffset


# %%
class regexLexer:
    # input: regex string
    # output: token stream
    def __init__(self, regexStr):
        self.regexStr = regexStr

    def lexer(self):
        tokenStream = []
        next_token_index = 0
        while next_token_index < len(self.regexStr):
            tokens, next_token_offset = getNextToken(self.regexStr[next_token_index:], tokenStream)
            next_token_index += next_token_offset
            tokenStream += tokens
            # print(next_token_index, next_token_offset)
        # for i in range(len(self.regexStr)):
        #     token = Token(getTypeToken(self.regexStr[i]), self.regexStr[i])
        #     tokenStream.append(token)
        return tokenStream


# regexLexer = regexLexer('a|b')
# tokenStream = regexLexer.lexer()
# for token in tokenStream:
#     print(token.ttype, token.content)


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
        print(' ' * indent + ('LAZY' if node.lazy else '') + 'STAR')
        print_ast(node.left, indent + 2)
    elif isinstance(node, PlusAstNode):
        print(' ' * indent + ('LAZY' if node.lazy else '') + 'PLUS')
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


# %%
## let's define a CFG for the language
# S -> E
# E -> T '|' E | T
# T -> C F T | C
# F -> '*' | '+' | '?' | epsilon
# C -> L | '(' E ')' | '[' L DASH L ']' | epsilon
# L -> LITERAL | ESCAPED
# OR -> '|' | epsilon
# STAR -> '*' | epsilon
# PLUS -> '+' | epsilon
# QUESTION_MARK -> '?' | epsilon
# OPEN_PAREN -> '(' | epsilon
# CLOSED_PAREN -> ')' | epsilon
# OPEN_SQUARE_BRACKET -> '[' | epsilon
# CLOSED_SQUARE_BRACKET -> ']' | epsilon
# DASH -> '-' | epsilon
# LITERAL -> any character except '|' '*', '+', '?', '(', ')', '[', ']', '\\', and '-'

class ParseRegex:
    def __init__(self, tokenStream):
        self.tokenStream = tokenStream
        self.currToken = 0

    def advance(self):
        self.currToken += 1

    def parse(self):
        ast = self.parse_E()
        if self.currToken < len(self.tokenStream):
            print(len(self.tokenStream))
            print(self.currToken)
            raise Exception("Unexpected token: ")

        return ast

    def parse_E(self):
        ast = self.parse_T()
        if self.match(TokenType.OR):
            left = ast
            right = self.parse_E()
            ast = OrAstNode(left, right)
        return ast

    def parse_T(self):
        ast = self.parse_C()
        if self.currToken < len(self.tokenStream):
            ttype = self.tokenStream[self.currToken].ttype
            # print(f'get a {ttype}')
            if ttype in [TokenType.LITERAL, TokenType.DIGITS, TokenType.ANY,
                         TokenType.OPEN_PAREN, TokenType.OPEN_SQUARE_BRACKET,
                         TokenType.DIGIT, TokenType.NOTDIGIT,
                         TokenType.SPACE, TokenType.NOTSPACE,
                         TokenType.WORD, TokenType.NOTWORD,
                         TokenType.BACK_REFERENCE]:
                left = ast
                right = self.parse_T()
                ast = SeqAstNode(left, right)
        return ast

    def parse_C(self):
        # DEBUG
        # print(in_sq_bracket)
        # print('in parse_C is ')
        type_now = self.tokenStream[self.currToken - 1].ttype
        # print(type_now)

        quantative_token_hit = False

        if self.match(TokenType.LITERAL):
            # DEBUG
            # print(self.tokenStream[self.currToken - 1].content)
            ast = LiteralCharacterAstNode(self.tokenStream[self.currToken - 1].content)
        elif self.match(TokenType.DIGITS):
            ast = LiteralCharacterAstNode(self.tokenStream[self.currToken - 1].content)
        elif self.match(TokenType.OPEN_PAREN):
            if self.match(TokenType.TURE_ASSERT):
                ast = self.parse_E()
                ast = AssertAstNode(ast, True)
            elif self.match(TokenType.FALSE_ASSERT):
                ast = self.parse_E()
                ast = AssertAstNode(ast, False)
            elif self.match(TokenType.NOT_GROUP):
                ast = self.parse_E()
                ast = NonCapturingGroupAstNode(ast)
            else:
                ast = self.parse_E()

            self.expect(TokenType.CLOSED_PAREN)

        elif self.match(TokenType.OPEN_SQUARE_BRACKET):
            negated = False
            if self.match(TokenType.NOT):
                negated = True
            clas, ranges = self.parse_L()
            self.expect(TokenType.CLOSED_SQUARE_BRACKET)
            ast = SquareBracketAstNode(clas)
            ast.ranges = ranges
            if negated:
                ast.negated = True
        elif self.match(TokenType.SPACE):
            ast = SquareBracketAstNode({' '})
        elif self.match(TokenType.DIGIT):
            # print('here is a \d')
            ast = SquareBracketAstNode({i for i in range(10)})
        elif self.match(TokenType.WORD):
            ast = SquareBracketAstNode(
                {chr(i) for i in range(65, 91)} | {chr(i) for i in range(97, 123)} | {i for i in range(10)})
        elif self.match(TokenType.NOTSPACE):
            ast = SquareBracketAstNode({chr(i) for i in range(33, 127)})
        elif self.match(TokenType.NOTDIGIT):
            ast = SquareBracketAstNode({chr(i) for i in range(33, 48)} | {chr(i) for i in range(58, 127)})
        elif self.match(TokenType.NOTWORD):
            ast = SquareBracketAstNode(
                {chr(i) for i in range(33, 65)} | {chr(i) for i in range(91, 97)} | {chr(i) for i in range(123, 127)})
        elif self.match(TokenType.BACK_REFERENCE):
            ast = BackReferenceAstNode(self.tokenStream[self.currToken - 1].content)
        # elif self.match(TokenType.ANY):
        #     ast = LiteralCharacterAstNode(ord('.'))
        elif self.match(TokenType.START):
            right = self.parse_E()
            # print(right.get_class_name())
            ast = STARTAstNode(right, right.get_class_name())

        else:
            print('unknown token: ', type_now)
            ast = AstNode()

        if self.match(TokenType.STAR):
            ast = StarAstNode(ast)
            quantative_token_hit = True
        elif self.match(TokenType.PLUS):
            ast = PlusAstNode(ast)
            quantative_token_hit = True

        elif self.match(TokenType.OPEN_BRACE):
            min_val = self.tokenStream[self.currToken - 1].content
            max_val = None
            if self.match(TokenType.CLOSED_BRACE):
                max_val = self.tokenStream[self.currToken - 1].content
                # DEBUG
                # print('min_val: ', min_val)
                # print('max_val: ', max_val)
            ast = QuantifierAstNode(ast, min_val, max_val)
            # print(ast.char)
            quantative_token_hit = True

        elif self.match(TokenType.QUESTION_MARK):
            ast = QuestionMarkAstNode(ast)

        if self.match(TokenType.QUESTION_MARK):
            assert quantative_token_hit, "Quantative token not hit"
            ast.lazy = True

        return ast

    def parse_L(self):
        clas = set()
        que = []
        range = []
        while self.currToken < len(self.tokenStream):
            ttype = self.tokenStream[self.currToken].ttype
            # print(ttype)
            if ttype == TokenType.CLOSED_SQUARE_BRACKET:
                break
            elif ttype == TokenType.LITERAL or ttype == TokenType.DIGITS:
                clas.add(self.tokenStream[self.currToken].content)
                que.append(self.tokenStream[self.currToken].content)

            elif ttype == TokenType.DASH:
                if len(clas) == 0 \
                        or self.currToken + 1 == len(self.tokenStream) \
                        or self.tokenStream[self.currToken + 1].ttype == TokenType.CLOSED_SQUARE_BRACKET:
                    clas.add('-')
                else:
                    # get last character in que
                    start = ord(chr(que.pop()))
                    clas.pop()
                    end = ord(chr(self.tokenStream[self.currToken + 1].content))
                    # print(chr(start), chr(end))
                    # for i in range(start, end + 1):
                    #    clas.add(chr(i))
                    range.append((start, end))
                    self.currToken += 1

            self.currToken += 1
        return clas, range

    def match(self, ttype):
        if self.currToken >= len(self.tokenStream):
            return False
        if self.tokenStream[self.currToken].ttype == ttype:
            self.currToken += 1
            return True
        return False

    def expect(self, ttype):
        if not self.match(ttype):
            raise Exception("Expected token")


# %% [markdown]
# # 测试需求1

# %%
import re


def is_valid_regex(regex):
    try:
        re.compile(regex)
        return True
    except re.error:
        return False


# %% [markdown]
# # 运行
import csv


def read_csv_to_regex_list(file_path):
    regex_list = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            regex_list.append(row[0])
    return regex_list


def collect_depth_limited_nodes(node, max_depth=99, current_depth=0):
    result = []
    # print(f'current_depth: {current_depth}')
    if current_depth >= max_depth:
        print("depth limit reached")
        return result

    if isinstance(node, (StarAstNode, PlusAstNode, QuestionMarkAstNode)):
        temp_node = node.left
        if isinstance(temp_node, LiteralCharacterAstNode):
            result.append(chr(temp_node.char))
            if isinstance(node, StarAstNode):
                result.append('*')
            elif isinstance(node, PlusAstNode):
                result.append('+')
            elif isinstance(node, QuestionMarkAstNode):
                result.append('?')
        else:
            temp_result = collect_depth_limited_nodes(temp_node, max_depth, current_depth + 1)
            if temp_result:
                temp_result = list(set(temp_result) - set(result))  # 确保 temp_result 是列表
                if isinstance(temp_node, SeqAstNode):
                    temp_result = ['(' + item + ')' for item in temp_result]
                elif isinstance(node, SquareBracketAstNode):
                    temp_result = ['[' + item + ']' for item in temp_result]
                if isinstance(node, PlusAstNode):
                    temp_result = [item + '+' for item in temp_result]  # 给所有元素添加 '+'
                elif isinstance(node, QuestionMarkAstNode):
                    temp_result = [item + '?' for item in temp_result]
                elif isinstance(node, StarAstNode):
                    temp_result = [item + '*' for item in temp_result]
                result.extend(temp_result)
        return result
    elif isinstance(node, STARTAstNode):
        temp_node = node.right
        # print(temp_node)
        if isinstance(temp_node, LiteralCharacterAstNode):
            result.append('^' + chr(temp_node.char))
        else:
            temp_result = collect_depth_limited_nodes(node.right, max_depth, current_depth + 1)
            # print(temp_result)
            if temp_result:
                result.append('^(' + temp_result[0] + ')')
                for item in temp_result[1:]:
                    result.append(item)
    elif isinstance(node, AssertAstNode):
        temp_node = node.child
        # print(temp_node.get_class_name())
        temp_result = collect_depth_limited_nodes(temp_node, max_depth, current_depth + 1)
        temp_result = list(set(temp_result) - set(result))
        if temp_result:
            print(temp_result)
            if node.is_positive:
                temp_result = ['(?=' + item + ')' for item in temp_result]
            else:
                temp_result = ['(?!' + item + ')' for item in temp_result]

            result.extend(temp_result)
            print(result)

    if isinstance(node, LiteralCharacterAstNode):
        # special_chars = ['.', '^', '$', '*', '+', '?', '|', '\\', '(', ')', '[', ']', '{', '}']
        # if chr(node.char) in special_chars:
        #     result.append('\\' + chr(node.char))
        # else:
        result.append(chr(node.char))
    elif isinstance(node, SeqAstNode):
        temp_result = []
        # print("get a seq node")
        # print(node.children)
        for child in node.children:
            # print(f'current_depth: {current_depth}, match a seq node')
            temp_result.extend(collect_depth_limited_nodes(child, max_depth, current_depth + 1))
            # print("get a child:")
            # print(child.get_class_name())
        # Merge consecutive literal characters
        merged_result = []
        temp_str = ''
        # print("temp_result: ", temp_result)
        for item in temp_result:
            if isinstance(item, str) and len(item) == 1:
                temp_str += item
            else:
                if temp_str:
                    merged_result.append(temp_str)
                    temp_str = ''
                merged_result.append(item)
        if temp_str:
            merged_result.append(temp_str)
        # print(result)
        # print(merged_result)
        result.extend(merged_result)
    elif isinstance(node, OrAstNode):
        left_result = collect_depth_limited_nodes(node.left, max_depth, current_depth + 1)
        right_result = collect_depth_limited_nodes(node.right, max_depth, current_depth + 1)
        result.append('(' + '|'.join(left_result + right_result) + ')')

    elif isinstance(node, NonCapturingGroupAstNode):
        result.append('?:')
        result.extend(collect_depth_limited_nodes(node.child, max_depth, current_depth + 1))
    elif isinstance(node, SquareBracketAstNode):
        bracket_content = []
        if node.negated:
            bracket_content.append('^')

        for start, end in node.ranges:
            bracket_content.append(chr(start) + '-' + chr(end))
        for char in node.clas:
            bracket_content.append(char)
        result.append('[' + ''.join(bracket_content) + ']')
    elif isinstance(node, BackReferenceAstNode):
        result.append('\\' + str(node.index))

    elif isinstance(node, QuantifierAstNode):
        quantifier_str = '{' + str(node.min) + (',' if node.max != 0 else '') + (
            str(node.max) if node.max > 0 and node.max != float('inf') else '') + '}'
        left_nodes = collect_depth_limited_nodes(node.left, max_depth, current_depth + 1)
        if left_nodes:
            result.append(left_nodes[0] + quantifier_str)
            result.extend(left_nodes[1:])
        else:
            result.append(quantifier_str)

    return result


def req1(regex):
    print('Req 1 : regex to NFA')
    regexlexer = regexLexer(regex)
    tokenStream = regexlexer.lexer()
    print('AST for regex: ', regex)
    parseRegex = ParseRegex(tokenStream)
    ## handle Exception
    throwException = False
    AST = parseRegex.parse()
    print_ast(AST)
    nodes = collect_depth_limited_nodes(AST)
    print('Nodes: ', nodes)
    return nodes


def process_string(input_string):
    # 找到第一个和第二个'/'的位置
    first_slash_index = input_string.find('/')
    second_slash_index = input_string.rfind('/')

    # 截取两个'/'中间的部分
    if first_slash_index != -1 and second_slash_index != -1:
        middle_part = input_string[first_slash_index + 1:second_slash_index]
    else:
        middle_part = ""

    # 保存'/'后的标志
    if second_slash_index != -1:
        after_slash = input_string[second_slash_index + 1:]
    else:
        after_slash = ""

    # 处理 `(^|&)` 结构
    middle_part = re.sub(r'(\(\^|\&\))', r'\\\1', middle_part)
    # 将非转义的字符 `.` 替换为字符集 `[^\n]`
    middle_part = re.sub(r'(?<!\\)\.', r'[^\n]', middle_part)

    return middle_part, after_slash


def main():
    file_path = 'pcre_fields.csv'
    regex_list = read_csv_to_regex_list(file_path)
    with open('result.txt', 'w', encoding='utf-8') as file:

        for regex in regex_list:
            regex, flags = process_string(regex)
            if not is_valid_regex(regex):
                print(f'Invalid regex: {regex}')
            else:
                node = req1(regex)
                file.write(f'regex : {regex}\n')
                file.write(f'nodes : {node}\n')

# if __name__ == "__main__":
#    main()

# regexTestBouns1='((((AB)|[A-Z])+)([A-Z]*))'
# regexTestBouns2='(((((ABE)|C)|((([A-Z])S)*))+)((AB)C))'
# regexTestBouns3='((([a-z_])(([a-z0-9_])*))(([!?])?))'
# regex = r'^a+?.{2}b'  # '(?=foo)bar'
# regex = r'http\x3A\x2f\x2f1\.usa\.gov\x2f[a-f0-9]{6,8}'
#  = r'abc|d\?'

# if not is_valid_regex(regex):
#   print('invalid regex compilation failed')
# else:
# req1(regex)





