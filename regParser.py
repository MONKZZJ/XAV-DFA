import re

from regLexer import TokenType, RegexLexer, Token
from astNodes import *

class RegexParser:
    def __init__(self, tokenStream):
        self.tokenStream = tokenStream

    def getNextNodeLength(self, stream: list[Token]) -> int:
        token = stream[0]
        temp_length = 1
        if token.ttype in [TokenType.LITERAL, TokenType.WORD, TokenType.NOTWORD, TokenType.DIGIT, TokenType.NOTDIGIT, TokenType.SPACE, TokenType.NOTSPACE, TokenType.ANY]:
            pass
        elif token.ttype == TokenType.OPEN_PAREN:
            level = 1
            for i in range(1, len(stream)):
                if stream[i].ttype == TokenType.OPEN_PAREN:
                    level += 1
                elif stream[i].ttype == TokenType.CLOSED_PAREN:
                    level -= 1
                    if level == 0:
                        temp_length = i + 1
                        break
            else:
                raise Exception("unbalanced parenthesis")
        elif token.ttype == TokenType.OPEN_SQUARE_BRACKET:
            level = 1
            for i in range(1, len(stream)):
                if stream[i].ttype == TokenType.OPEN_SQUARE_BRACKET:
                    level += 1
                elif stream[i].ttype == TokenType.CLOSED_SQUARE_BRACKET:
                    level -= 1
                    if level == 0:
                        temp_length = i + 1
                        break
            else:
                raise Exception("unbalanced square brackets")
        else:
            raise Exception(f"token type {token.ttype} should not be here")
        ## check quantifier
        if temp_length < len(stream):
            next_token = stream[temp_length]
            if next_token.ttype == TokenType.QUESTION_MARK:
                temp_length += 1
            else:
                # check multiple quantifiers
                if next_token.ttype in [TokenType.STAR, TokenType.PLUS]:
                    temp_length += 1
                elif next_token.ttype == TokenType.OPEN_BRACE:
                    assert stream[temp_length+1] == TokenType.CLOSED_BRACE, "invalid quantifier"
                    temp_length += 2
                # check lazy mark
                if temp_length < len(stream):
                    next_token = stream[temp_length]
                    if next_token.ttype == TokenType.QUESTION_MARK:
                        temp_length += 1
        return temp_length


    def parseStream(self, stream: list[Token]=None) -> SeqAstNode:
        if stream is None:
            stream = self.tokenStream
        root = SeqAstNode()
        stream_pointer = 0
        while stream_pointer < len(stream):
            token = stream[stream_pointer]
            stream_pointer += 1
            if token.ttype == TokenType.OR:
                assert root.children.length > 0, "OR must have left children"
                lastNode  = root.children[-1]
                nextNodeLength = self.getNextNodeLength(stream[stream_pointer:])
                nextNode = self.parseStream(stream[stream_pointer:stream_pointer+nextNodeLength])
                root.children[-1] = OrAstNode(lastNode, nextNode)
                stream_pointer += nextNodeLength
            elif token.ttype in [TokenType.STAR, TokenType.PLUS, TokenType.QUESTION_MARK, TokenType.OPEN_BRACE]:
                nextOffset = 0
                assert len(root.children) > 0, "quantifier must have left children"
                lastNode = root.children[-1]
                assert lastNode.quantative == False, "last token have quantifier"
                lastNode.quantative = True
                if token.ttype == TokenType.QUESTION_MARK:
                    lastNode.min_repeat = 0
                    lastNode.max_repeat = 1
                else:
                    if token.ttype == TokenType.STAR:
                        lastNode.min_repeat = 0
                        lastNode.max_repeat = -1
                    elif token.ttype == TokenType.PLUS:
                        lastNode.min_repeat = 1
                        lastNode.max_repeat = -1
                    elif token.ttype == TokenType.OPEN_BRACE:
                        assert stream[stream_pointer].ttype == TokenType.CLOSED_BRACE, "invalid quantifier"
                        lastNode.min_repeat = int(token.content)
                        lastNode.max_repeat = int(stream[stream_pointer].content)
                        nextOffset = 1
                    # check lazy mark 
                    if stream_pointer + nextOffset < len(stream) and stream[stream_pointer + nextOffset].ttype == TokenType.QUESTION_MARK:
                        lastNode.lazy = True
                        nextOffset += 1
                stream_pointer += nextOffset
            elif token.ttype == TokenType.OPEN_PAREN:
                assert stream[0].ttype not in [TokenType.TURE_ASSERT, TokenType.FALSE_ASSERT, TokenType.NOT_GROUP], "assertion not supported"
                nextOffset = 0
                depth = 1
                for i in range(stream_pointer, len(stream)):
                    if stream[i].ttype == TokenType.OPEN_PAREN:
                        depth += 1
                    elif stream[i].ttype == TokenType.CLOSED_PAREN:
                        depth -= 1
                        if depth == 0:
                            nextOffset = i + 1
                            break                    
                else:
                    raise Exception("unbalanced parenthesis")
                root.children.append(self.parseStream(stream[stream_pointer:nextOffset-1]))
                stream_pointer = nextOffset
            elif token.ttype == TokenType.OPEN_SQUARE_BRACKET:
                reversed = False
                if stream[stream_pointer].ttype == TokenType.NOT:
                    reversed = True
                    stream_pointer += 1
                for i in range(stream_pointer, len(stream)):
                    if stream[i].ttype == TokenType.CLOSED_SQUARE_BRACKET:
                        nextOffset = i + 1
                        break
                else:
                    raise Exception("unbalanced square brackets")
                node = SquareBracketAstNode(self.parseStream(stream[stream_pointer:nextOffset-1]).children, reversed)
                root.children.append(node)
                stream_pointer = nextOffset
            elif token.ttype in [TokenType.NOTWORD, TokenType.WORD, TokenType.NOTSPACE, TokenType.SPACE, TokenType.NOTDIGIT, TokenType.DIGIT]:
                root.children.append(LiteralCharacterAstNode(token.ttype))
                if token.ttype in [TokenType.WORD, TokenType.SPACE, TokenType.DIGIT]:
                    node = LiteralSpecialCharacterAstNode(False, token.ttype==TokenType.WORD, token.ttype==TokenType.SPACE, token.ttype==TokenType.DIGIT)
                elif token.ttype in [TokenType.NOTWORD, TokenType.NOTSPACE, TokenType.NOTDIGIT]:
                    node = LiteralSpecialCharacterAstNode(True, token.ttype==TokenType.NOTWORD, token.ttype==TokenType.NOTSPACE, token.ttype==TokenType.NOTDIGIT)
                root.children.append(node)
            elif token.ttype == TokenType.START:
                assert len(root.children) == 0, "start must have no children"
                root.start = True
            elif token.ttype == TokenType.END:
                assert stream_pointer == len(stream), "end must be the last token"
                root.end = True
            elif token.ttype == TokenType.ANY:
                # Equivalent to [^\n\r].
                node = SquareBracketAstNode([LiteralCharacterAstNode(ord('\n')), LiteralCharacterAstNode(ord('\r'))], True)
                root.children.append(node)
            elif token.ttype == TokenType.LITERAL:
                root.children.append(LiteralCharacterAstNode(token.content))
            else:
                raise NotImplementedError(f"token type {token.ttype} not implemented")
        return root

def is_valid_regex(regex):
    raise NotImplementedError("unchecked")
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
    raise NotImplementedError("unchecked")
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
    raise NotImplementedError("unchecked")
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
    raise NotImplementedError("unchecked")
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
    parseRegex = RegexParser(tokenStream)
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





