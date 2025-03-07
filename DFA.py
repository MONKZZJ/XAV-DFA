from parser import ParseRegex, regexLexer, TokenType, collect_depth_limited_nodes,\
    AstNode, STARTAstNode, OrAstNode, SeqAstNode, StarAstNode, PlusAstNode, QuestionMarkAstNode, \
    LiteralCharacterAstNode, SquareBracketAstNode, QuantifierAstNode, AssertAstNode, \
    BackReferenceAstNode, NonCapturingGroupAstNode
import csv

class DFA:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.transitions = {}
        self.start_state = None
        self.accept_states = set()

    def add_state(self, state, is_accept=False):
        self.states.add(state)
        if is_accept:
            self.accept_states.add(state)

    def set_start_state(self, state):
        self.start_state = state

    def add_transition(self, from_state, input_char, to_state):
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][input_char] = to_state
        self.alphabet.add(input_char)

    def is_accepting(self, state):
        return state in self.accept_states

    def get_next_state(self, current_state, input_char):
        return self.transitions.get(current_state, {}).get(input_char)

    def accepts(self, input_string):
        current_states = self.epsilon_closure({self.start_state})
        for char in input_string:
            next_states = set()
            for state in current_states:
                next_state = self.get_next_state(state, char)
                if next_state is not None:
                    next_states.update(self.epsilon_closure({next_state}))

            current_states = next_states
            if not current_states:
                return False
        return any(self.is_accepting(state) for state in current_states)

    def epsilon_closure(self, states):
        # 计算给定状态的 ε-闭包
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            epsilon_state = self.get_next_state(state, '')
            if epsilon_state is not None and epsilon_state not in closure:
                closure.add(epsilon_state)
                stack.append(epsilon_state)
        return closure

    def print_dfa(self):
        print("States:", self.states)
        print("Alphabet:", self.alphabet)
        print("Transitions:")
        for state, transitions in self.transitions.items():
            for char, next_state in transitions.items():
                if char:
                    print(f"  {state} --{ord(char)}--> {next_state}")
                else:
                    print(f"  {state} -- ε --> {next_state}")
        print("Start state:", self.start_state)
        print("Accept states:", self.accept_states)

    def to_transition_table(self):
        table = [[-1 for _ in range(256)] for _ in range(256)]
        for from_state, transitions in self.transitions.items():
            for char, to_state in transitions.items():
                if char:
                    table[ord(char)][from_state] = to_state
        return table


def regex_to_dfa(regex):

    dfa = DFA()
    state_counter = 0

    def traverse(node, current_state):
        nonlocal state_counter
        if isinstance(node, STARTAstNode):
            next_state = state_counter
            state_counter += 1
            dfa.add_state(next_state)
            dfa.set_start_state(next_state)
            return traverse(node.right, next_state)
        if isinstance(node, LiteralCharacterAstNode):
            if chr(node.char) == '.':
                next_state = state_counter
                state_counter += 1
                dfa.add_state(next_state)
                for char in range(256):
                    dfa.add_transition(current_state, chr(char), next_state)
                return next_state
            else:
                # print('no any')
                next_state = state_counter
                state_counter += 1
                dfa.add_state(next_state)
                dfa.add_transition(current_state, chr(node.char), next_state)
                return next_state
        elif isinstance(node, SeqAstNode):
            for child in node.children:
                current_state = traverse(child, current_state)
            return current_state
        elif isinstance(node, OrAstNode):
            left_state = traverse(node.left, current_state)
            right_state = traverse(node.right, current_state)
            return max(left_state, right_state)
        elif isinstance(node, StarAstNode):
            next_state = state_counter
            state_counter += 1
            dfa.add_state(next_state)
            dfa.add_transition(current_state, '', next_state)
            inner_state = traverse(node.left, next_state)
            dfa.add_transition(inner_state, '', next_state)
            return next_state
        elif isinstance(node, PlusAstNode):
            next_state = traverse(node.left, current_state)
            dfa.add_transition(next_state, '', current_state)
            return next_state
        elif isinstance(node, QuestionMarkAstNode):
            next_state = state_counter
            state_counter += 1
            dfa.add_state(next_state)
            dfa.add_transition(current_state, '', next_state)
            traverse(node.left, current_state)
            return next_state
        elif isinstance(node, SquareBracketAstNode):
            next_state = state_counter
            state_counter += 1
            dfa.add_state(next_state)
            # print(node.clas)
            for char in node.clas:
                dfa.add_transition(current_state, chr(char), next_state)
            return next_state
        elif isinstance(node, QuantifierAstNode):
            min_state = traverse(node.left, current_state)
            for _ in range(node.min - 1):
                min_state = traverse(node.left, min_state)
            if node.max is None:
                dfa.add_transition(min_state, '', current_state)
            else:
                for _ in range(node.max - node.min):
                    min_state = traverse(node.left, min_state)
            return min_state
        elif isinstance(node, AssertAstNode):
            return traverse(node.child, current_state)
        elif isinstance(node, BackReferenceAstNode):
            return current_state
        elif isinstance(node, NonCapturingGroupAstNode):
            return traverse(node.child, current_state)
        else:
            raise ValueError("Unknown AST node type")

    start_state = state_counter
    state_counter += 1
    dfa.add_state(start_state)
    dfa.set_start_state(start_state)
    final_state = traverse(ast, start_state)
    dfa.add_state(final_state, is_accept=True)

    return dfa

def precess_re(regex):
    lexer = regexLexer(regex)
    tokenStream = lexer.lexer()
    parser = ParseRegex(tokenStream)
    ast = parser.parse()
    return ast

def save_transition_table_to_file(trans_table, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in trans_table:
            writer.writerow(row)

if __name__ == "__main__":
    regex = r'^a+?.{2}b'


    ast = precess_re(regex)
    nodes = collect_depth_limited_nodes(ast)
    dfa = regex_to_dfa(ast)
    dfa.print_dfa()

    trans_table = dfa.to_transition_table()
    save_transition_table_to_file(trans_table, 'transition_table.csv')
    #for row in trans_table:
    #   print(row)

    test_string = 'acdb'
    print(f"The string '{test_string}' is accepted by the DFA: {dfa.accepts(test_string)}")