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
from enum import Enum,auto
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
    def __init__(self, ttype, content:int = -1):
        
        self.ttype = ttype
        self.content = content
    def __repr__(self) -> str:
        return f"Token({self.ttype=}, {self.content=})"
        
def getNextToken(string: str, tokenStream:list[Token]) -> tuple[list[Token], int]:
    """
    获取下一个标记

    参数:
    string (str): 输入字符串
    tokenStream (list[Token]): 标记流

    返回:
    tuple[list[Token], int]: 返回标记列表和下一个偏移量
    """
    global in_PAREN, in_sq_bracket, in_BR
    token = string[0]
    nextOffset = 1
    returnTokens = []
    if in_sq_bracket:
        if token == ']':
            in_sq_bracket = False
            returnTokens.append(Token(TokenType.CLOSED_SQUARE_BRACKET, -1))
        elif token == '-':
            returnTokens.append(Token(TokenType.DASH, -1))
        else:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
    elif token == '|':
        returnTokens.append(Token(TokenType.OR, -1))
    elif token == '.':
        returnTokens.append(Token(TokenType.ANY, -1))
    elif token == '*':
        returnTokens.append(Token(TokenType.STAR, -1))
    elif token == '+':
        returnTokens.append(Token(TokenType.PLUS, -1))
    elif token == '?':
        returnTokens.append(Token(TokenType.QUESTION_MARK, -1))
    elif token == '(':
        # todo: combine to one token?
        returnTokens.append(Token(TokenType.OPEN_PAREN, -1))
        if string[1] == '?':
            if string[2] == '=':
                returnTokens.append(Token(TokenType.TURE_ASSERT, -1))
                nextOffset = 3
            elif string[2] == '!':
                returnTokens.append(Token(TokenType.FALSE_ASSERT, -1))
                nextOffset = 3
            elif string[2] == ":":
                returnTokens.append(Token(TokenType.NOT_GROUP, -1))
                nextOffset = 3
            else:
                raise ValueError("Invalid assertion,expected '=' or '!'")
    elif token == ')':
        returnTokens.append(Token(TokenType.CLOSED_PAREN, -1))
    elif token == '[':
        in_PAREN += 1
        in_sq_bracket = True
        returnTokens.append(Token(TokenType.OPEN_SQUARE_BRACKET, -1))
        if string[1] == '^':
            returnTokens.append(Token(TokenType.NOT, -1))
            nextOffset = 2
    elif token == ']':
        returnTokens.append(Token(TokenType.LITERAL, ord(']')))
    elif token == '-':
        returnTokens.append(Token(TokenType.LITERAL, ord('-')))
    elif token == '^':
        returnTokens.append(Token(TokenType.START, -1))
    elif token == '{':
        # 检查是否匹配量词格式 ^{(\d+),(\d*?)}
        quantifier_match = re.match(r'^\{(\d+),(\d*?)\}', string)
        if quantifier_match:
            min_val = int(quantifier_match.group(1))
            max_val = int(quantifier_match.group(2)) if quantifier_match.group(2) else float('inf')
            nextOffset = quantifier_match.end()
            returnTokens.append(Token(TokenType.OPEN_BRACE, min_val))
            returnTokens.append(Token(TokenType.CLOSED_BRACE, max_val))
        else:
            returnTokens.append(Token(TokenType.LITERAL, ord(token)))
    elif token == '\\':
        char = string[1]    
        nextOffset = 2
        if char.isdigit():
            octal_match = re.match(r'^[0-7]{1,3}', string[1:])
            if octal_match is not None:
                matchstr = octal_match.group()    
                if int(matchstr, base=8) > 255:
                    matchstr = matchstr[:-1]
                returnTokens.append(Token(TokenType.LITERAL, int(matchstr, base=8)))
                nextOffset = len(matchstr) + 1
            else:
                returnTokens.append(Token(TokenType.BACK_REFERENCE, int(char)))
        elif char in ["+", "*", "?", "^", "$", "\\", "." , "-", "[", "]", "{", "}", "(", ")", "|", "/", "t", "n", "r", "v", "f", "b", "a", "e"]:
            #  get ascii code 
            returnTokens.append(Token(TokenType.LITERAL, ord(char)))
        elif char == "x":
            hex_match = re.match(r'^[0-9a-fA-F]{1,2}', string[2:]).group()
            returnTokens.append(Token(TokenType.LITERAL, int(hex_match, base=16)))
            nextOffset = len(hex_match) + 2
        elif char == "u":
            raise NotImplementedError("Unicode escape sequence not implemented")
            # 太复杂，暂不实现
            # if string[2] == "{":
            #     end_index = string.find("}")
            #     assert end_index != -1, "Invalid unicode escape sequence"
            #     assert (end_index - 3) % 2 == 0, "Invalid unicode escape sequence"
            #     for i in range(3, end_index, 2):
            #         returnTokens.append(Token(TokenType.LITERAL, int(string[i:i+2], base=16)))
            #     nextOffset = end_index + 1                    
            # else:
            #     returnTokens.append(Token(TokenType.LITERAL, int(string[2:4], base=16)))
            #     returnTokens.append(Token(TokenType.LITERAL, int(string[4:6], base=16)))
            #     nextOffset = 6
        elif char == "c":
            assert ord("A") <= ord(string[2]) <= ord("Z") or ord("a") <= ord(string[2]) <= ord("z"), "Invalid control character"
            if ord(string[2]) < "a":
                returnTokens.append(Token(TokenType.LITERAL, ord(string[2]) - ord("A") + 1))
            else: 
                returnTokens.append(Token(TokenType.LITERAL, ord(string[2]) - ord("a") + 1))
            nextOffset = 3
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
            raise ValueError(f"unsupported escape character: {char}")        
    else:
        returnTokens.append(Token(TokenType.LITERAL, ord(token)))
    return returnTokens, nextOffset

# %%
class RegexLexer:
    """
    正则表达式词法分析器

    输入: 正则表达式字符串
    输出: 标记流
    """
    def __init__(self, regexStr):
        self.regexStr = regexStr
    def lexer(self):
        """
        词法分析器函数，将正则表达式字符串转换为标记流

        返回:
        list[Token]: 标记流
        """
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
    

from astNodes import *

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
        """
        前进到下一个标记
        """
        self.currToken += 1

    def parse(self):
        """
        解析标记流并生成AST

        返回:
        AstNode: AST根节点
        """
        ast = self.parse_E()
        if self.currToken < len(self.tokenStream):
            print(len(self.tokenStream))
            print(self.currToken)
            raise Exception("Unexpected token: ")

        return ast

    def parse_E(self):
        """
        解析E规则

        返回:
        AstNode: AST节点
        """
        ast = self.parse_T()
        if self.match(TokenType.OR):
            left = ast
            right = self.parse_E()
            ast = OrAstNode(left, right)
        return ast

    def parse_T(self):
        """
        解析T规则

        返回:
        AstNode: AST节点
        """
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
        """
        解析C规则

        返回:
        AstNode: AST节点
        """
        # DEBUG
        # print(in_sq_bracket)
        # print('in parse_C is ')
        type_now = self.tokenStream[self.currToken - 1].ttype
        # print(type_now)

        quantative_token_hit = False

        if self.match(TokenType.LITERAL):
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
            ast = SquareBracketAstNode({chr(i) for i in range(65, 91)} | {chr(i) for i in range(97, 123)} | {i for i in range(10)})
        elif self.match(TokenType.NOTSPACE):
            ast = SquareBracketAstNode({chr(i) for i in range(33, 127)})
        elif self.match(TokenType.NOTDIGIT):
            ast = SquareBracketAstNode({chr(i) for i in range(33, 48)} | {chr(i) for i in range(58, 127)})
        elif self.match(TokenType.NOTWORD):
            ast = SquareBracketAstNode({chr(i) for i in range(33, 65)} | {chr(i) for i in range(91, 97)} | {chr(i) for i in range(123, 127)})
        elif self.match(TokenType.BACK_REFERENCE):
            ast = BackReferenceAstNode(self.tokenStream[self.currToken - 1].content)
        elif self.match(TokenType.ANY):
            ast = LiteralCharacterAstNode(ord('.'))
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
        """
        解析L规则

        返回:
        tuple[set, list]: 字符集合和范围列表
        """
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
                    end = ord(chr(self.tokenStream[self.currToken + 1].content))
                    # print(chr(start), chr(end))
                    # for i in range(start, end + 1):
                    #    clas.add(chr(i))
                    range.append((start, end))
                    self.currToken += 1


            self.currToken += 1
        return clas, range

    def match(self, ttype):
        """
        匹配当前标记类型

        参数:
        ttype (TokenType): 标记类型

        返回:
        bool: 是否匹配
        """
        if self.currToken >= len(self.tokenStream):
            return False
        if self.tokenStream[self.currToken].ttype == ttype:
            self.currToken += 1
            return True
        return False

    def expect(self, ttype):
        """
        期望匹配当前标记类型

        参数:
        ttype (TokenType): 标记类型

        抛出:
        Exception: 如果不匹配则抛出异常
        """
        if not self.match(ttype):
            raise Exception("Expected token")

# %% [markdown]
# # 测试需求1

# %%
import re


def is_valid_regex(regex):
    """
    检查正则表达式是否有效

    参数:
    regex (str): 正则表达式

    返回:
    bool: 是否有效
    """
    try:
        re.compile(regex)
        return True
    except re.error:
        return False

# %% [markdown]
# # 运行
import csv

def read_csv_to_regex_list(file_path):
    """
    从CSV文件读取正则表达式列表

    参数:
    file_path (str): CSV文件路径

    返回:
    list[str]: 正则表达式列表
    """
    regex_list = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            regex_list.append(row[0])
    return regex_list

def collect_depth_limited_nodes(node, max_depth=4, current_depth=0):
    """
    收集深度限制的节点

    参数:
    node (AstNode): AST节点
    max_depth (int): 最大深度
    current_depth (int): 当前深度

    返回:
    list: 节点列表
    """
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

    if isinstance(node, LiteralCharacterAstNode):
        result.append(chr(node.char))
    elif isinstance(node, SeqAstNode):
        temp_result = []
        for child in node.children:
            # print(f'current_depth: {current_depth}, match a seq node')
            temp_result.extend(collect_depth_limited_nodes(child, max_depth, current_depth + 1))
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
        result.extend(collect_depth_limited_nodes(node.left, max_depth, current_depth + 1))
        result.extend(collect_depth_limited_nodes(node.right, max_depth, current_depth + 1))
    elif isinstance(node, AssertAstNode):
        if node.is_positive:
            result.append('?=')
        else:
            result.append('?!')
        result.extend(collect_depth_limited_nodes(node.child, max_depth, current_depth + 1))
    elif isinstance(node, NonCapturingGroupAstNode):
        result.append('?:')
        result.extend(collect_depth_limited_nodes(node.child, max_depth, current_depth + 1))
    elif isinstance(node, SquareBracketAstNode):
        if node.ranges:
            for start, end in node.ranges:
                result.append('[' + chr(start) + '-' + chr(end) + ']')
        else:
            result.append('[' + ''.join(chr(c) if isinstance(c, int) else c for c in node.clas) + ']')
    elif isinstance(node, BackReferenceAstNode):
        result.append('\\' + str(node.index))

    elif isinstance(node, QuantifierAstNode):
        quantifier_str = '{' + str(node.min) + (',' if node.max != 0 else '') + (str(node.max) if node.max > 0 and node.max != float('inf') else '') + '}'
        left_nodes = collect_depth_limited_nodes(node.left, max_depth, current_depth + 1)
        if left_nodes:
            result.append(left_nodes[0] + quantifier_str)
            result.extend(left_nodes[1:])
        else:
            result.append(quantifier_str)

    return result
def req1(regex):
    """
    需求1：正则表达式转换为NFA

    参数:
    regex (str): 正则表达式

    返回:
    list: 节点列表
    """
    print('Req 1 : regex to NFA')
    regexlexer = RegexLexer(regex)
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
    """
    处理输入字符串，提取中间部分和斜杠后的标志

    参数:
    input_string (str): 输入字符串

    返回:
    tuple[str, str]: 中间部分和斜杠后的标志
    """
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

    return middle_part, after_slash

def main():
    """
    主函数，读取CSV文件并处理正则表达式
    """
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





if __name__ == "__main__":
    regex = r'^a+?.*.{2}b'  # '(?=foo)bar'

    if not is_valid_regex(regex):
        print('invalid regex compilation failed')
    else:
        req1(regex)

# regexTestBouns1='((((AB)|[A-Z])+)([A-Z]*))'
# regexTestBouns2='(((((ABE)|C)|((([A-Z])S)*))+)((AB)C))'
# regexTestBouns3='((([a-z_])(([a-z0-9_])*))(([!?])?))'





