import re 

in_PAREN = 0
in_sq_bracket = 0
in_BR = False

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
    