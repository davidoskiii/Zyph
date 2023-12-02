#######################################
# IMPORTS
#######################################

from string_with_arrows import *

import string
import os
import math
import random


#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

def run_script(fn):

    if not (isinstance(fn, str)):
        raise

    if fn.endswith('.zys'):
        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            print(e)

        _, error = run(fn, script)

        if error:
                print(f"Failed to finish executing script \"{fn}\"\n" + error.as_string())

        return RTResult().success(Number.null)

    else:
        print(f"Script file name must ends with \".zys\"\n" + f"But got '{fn}'")

#######################################
# ERRORS
#######################################

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result  = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        result  = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT			= 'INT'
TT_FLOAT    	= 'FLOAT'
TT_IDENTIFIER	= 'IDENTIFIER'
TT_KEYWORD		= 'KEYWORD'
TT_STRING       = 'STRING'
TT_PLUS     	= 'PLUS'
TT_MINUS    	= 'MINUS'
TT_MUL      	= 'MUL'
TT_DIV      	= 'DIV'
TT_POW		    = 'POW'
TT_EQ		    = 'EQ'
TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'
TT_LCURLY   	= 'LCURLY'
TT_RCURLY   	= 'RCURLY'
TT_LSQUARE   	= 'LSQUARE'
TT_RSQUARE   	= 'RSQUARE'
TT_ARROW        = 'ARROW'
TT_COMMA        = 'COMMA'
TT_EOF			= 'EOF'
TT_EE			= 'EE'
TT_NE			= 'NE'
TT_LT			= 'LT'
TT_GT			= 'GT'
TT_LTE			= 'LTE'
TT_GTE			= 'GTE'
TT_EOF			= 'EOF'
TT_MOD          = 'MOD'
TT_COLON        = 'COLON'
TT_NEWLINE      = 'NEWLINE'

KEYWORDS = [
	'var',
    'and',
    'or',
    'not',
    'if',
    'elif',
    'else',
    'for',
    'to',
    'step',
    'while',
    'function',
    'return',
    'continue',
    'break',
    'try',
    'catch'
]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'

#######################################
# LEXER
#######################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(self.make_minus_or_arrow())
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                self.make_comment_or_div(tokens)
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '%':
                tokens.append(Token(TT_MOD, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '{':
                tokens.append(Token(TT_LCURLY, pos_start=self.pos))
                self.advance()
            elif self.current_char == '}':
                tokens.append(Token(TT_RCURLY, pos_start=self.pos))
                self.advance()
            elif self.current_char == '[':
                tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == ']':
                tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
                self.advance()
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.current_char == ':':
                tokens.append(Token(TT_COLON, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
                self.advance()
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
                self.advance()
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
            num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()

        escape_characters = {
            'n': '\n',
            't': '\t'
        }

        while self.current_char != None and (self.current_char != '"' or escape_character):
            if escape_character:
                string += escape_characters.get(self.current_char, self.current_char)
            else:
                if self.current_char == '\\':
                    escape_character = True
                else:
                    string += self.current_char
            self.advance()
            escape_character = False

        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)


    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_minus_or_arrow(self):
        tok_type = TT_MINUS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = TT_ARROW

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = TT_EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_EE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_LTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TT_GTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def skip_comment(self):
        self.advance()

        while self.current_char != '\n':
            self.advance()

        self.advance()

    def make_comment_or_div(self, tokens):
        pos_start = self.pos.copy()
        self.advance()  # Skip '/'

        if self.current_char == '/':
            self.skip_comment()
        else:
            self.advance()
            tokens.append(Token(TT_DIV, pos_start=self.pos))


#######################################
# NODES
#######################################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class StringNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes

        self.pos_start = pos_start
        self.pos_end = pos_end

class DictionaryNode:
    def __init__(self, key_value_pairs, pos_start, pos_end):
        self.key_value_pairs = key_value_pairs
        self.pos_start = pos_start
        self.pos_end = pos_end

class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end

class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.should_auto_return = should_auto_return

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end

class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end

class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return

        self.pos_start = pos_start
        self.pos_end = pos_end

class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end

class TryCatchNode:
    def __init__(self, try_body, catch_body):
        self.try_body = try_body
        self.catch_body = catch_body
        self.pos_start = self.try_body.pos_start
        self.pos_end = self.catch_body.pos_end



#######################################
# PARSE RESULT
#######################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

#######################################
# PARSER
#######################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        self.update_current_tok()
        return self.current_tok

    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.current_tok

    def update_current_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '/', '%' or '^'"
            ))
        return res

    ###################################

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(ListNode(
            statements,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.matches(TT_KEYWORD, 'return'):
            res.register_advancement()
            self.advance()

            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'continue'):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'break'):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'return', 'continue', 'break', 'var', 'if', 'for', 'while', 'function', int, float, identifier, '+', '-', '(', '[', 'or', 'and' or 'not'"
            ))
        return res.success(expr)

    def try_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'try'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'try'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        try_body = res.register(self.statements())
        if res.error: return res

        if not self.current_tok.type == TT_RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.matches(TT_KEYWORD, 'catch'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "The try statement is incomplete, expected 'catch'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))
        res.register_advancement()
        self.advance()

        catch_body = res.register(self.statements())
        if res.error: return res

        if not self.current_tok.type == TT_RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))

        res.register_advancement()
        self.advance()

        return res.success(TryCatchNode(try_body, catch_body))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases('if'))
        if res.error: return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case))

    def if_expr_b(self):
        return self.if_expr_cases('elif')

    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(TT_KEYWORD, 'else'):
            res.register_advancement()
            self.advance()

            if not self.current_tok.type == TT_LCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '{'"
                ))

            res.register_advancement()
            self.advance()

            if self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)

                if self.current_tok.type == TT_RCURLY:
                    res.register_advancement()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '}'"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error: return res
                else_case = (expr, False)

                if not self.current_tok.type == TT_RCURLY:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '}'"
                    ))

                res.register_advancement()
                self.advance()

        return res.success(else_case)

    def if_expr_b_or_c(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.current_tok.matches(TT_KEYWORD, 'elif'):
            all_cases = res.register(self.if_expr_b())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error: return res

        return res.success((cases, else_case))

    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(TT_KEYWORD, case_keyword):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{case_keyword}'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '('"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.type == TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ')'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            if res.error: return res
            cases.append((condition, statements, True))

            if self.current_tok.type == TT_RCURLY:
                res.register_advancement()
                self.advance()
                if self.current_tok.type == TT_RCURLY:
                    res.register_advancement()
                    self.advance()
                else:
                    all_cases = res.register(self.if_expr_b_or_c())
                    if res.error: return res
                    new_cases, else_case = all_cases
                    cases.extend(new_cases)
            else:
                all_cases = res.register(self.if_expr_b_or_c())
                if res.error: return res
                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            expr = res.register(self.statement())
            if res.error: return res
            cases.append((condition, expr, False))

            if not self.current_tok.type == TT_RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            all_cases = res.register(self.if_expr_b_or_c())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        return res.success((cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'for'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'for'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '('"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier"
            ))

        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '='"
            ))

        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.type != TT_COMMA:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ','"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.matches(TT_KEYWORD, 'to'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'to'"
            ))

        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.type == TT_COMMA:

            res.register_advancement()
            self.advance()

            if not self.current_tok.matches(TT_KEYWORD, 'step'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected 'step'"
                ))

            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None

        if not self.current_tok.type == TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ')'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.type == TT_RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        body = res.register(self.statement())
        if res.error: return res

        if not self.current_tok.type == TT_RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))

        res.register_advancement()
        self.advance()

        return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'while'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'while'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '('"
            ))


        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.type == TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ')'"
            ))

        res.register_advancement()
        self.advance()

        if not self.current_tok.type == TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.type == TT_RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(WhileNode(condition, body, True))

        body = res.register(self.statement())
        if res.error: return res

        if not self.current_tok.type == TT_RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))

        res.register_advancement()
        self.advance()

        return res.success(WhileNode(condition, body, False))

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        if tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))

        elif tok.type == TT_LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)

        elif tok.type == TT_LCURLY:
            dictionary_expr = res.register(self.dictionary_expr())
            if res.error: return res
            return res.success(dictionary_expr)

        elif tok.matches(TT_KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, 'try'):
            try_expr = res.register(self.try_expr())
            if res.error: return res
            return res.success(try_expr)

        elif tok.matches(TT_KEYWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(TT_KEYWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif tok.matches(TT_KEYWORD, 'function'):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int, float, identifier, '+', '-' or '('"
        ))

    def dictionary_expr(self):
        res = ParseResult()
        key_value_pairs = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_RCURLY:
            res.register_advancement()
            self.advance()
        else:
            key_node = res.register(self.expr())
            if res.error:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[', '{' or 'NOT'"
                ))

            if self.current_tok.type != TT_COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':'"
                ))

            res.register_advancement()
            self.advance()

            value_node = res.register(self.expr())
            if res.error:
                return res

            key_value_pairs.append((key_node, value_node))

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                key_node = res.register(self.expr())
                if res.error: return res

                if self.current_tok.type != TT_COLON:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ':'"
                    ))

                res.register_advancement()
                self.advance()

                value_node = res.register(self.expr())
                if res.error: return res

                key_value_pairs.append((key_node, value_node))

            if self.current_tok.type != TT_RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ',' or '}'"
                ))

            res.register_advancement()
            self.advance()

        return res.success(DictionaryNode(
            key_value_pairs,
            pos_start,
            self.current_tok.pos_end.copy()
        ))


    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_LSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '['"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_RSQUARE:
            res.register_advancement()
            self.advance()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
                ))

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.current_tok.type != TT_RSQUARE:
                return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ',' or ']'"
                ))

            res.register_advancement()
            self.advance()

        return res.success(ListNode(
            element_nodes,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'function'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'function'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_IDENTIFIER:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '('"
                ))
        else:
            var_name_tok = None
            if self.current_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or '('"
                ))

        res.register_advancement()
        self.advance()
        arg_name_toks = []

        if self.current_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.current_tok)
            res.register_advancement()
            self.advance()

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected identifier"
                    ))

                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()

            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' or ')'"
                ))
        else:
            if self.current_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or ')'"
                ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_LCURLY:
            res.register_advancement()
            self.advance()

            if self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

                body = res.register(self.statements())
                if res.error: return res

                if not self.current_tok.type == TT_RCURLY:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '}'"
                    ))

                res.register_advancement()
                self.advance()

                return res.success(FuncDefNode(
                    var_name_tok,
                    arg_name_toks,
                    body,
                    True
                ))

            body = res.register(self.expr())
            if res.error: return res

            if self.current_tok.type != TT_RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(FuncDefNode(
                var_name_tok,
                arg_name_toks,
                body,
                False
            ))
        else: return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

    def power(self):
        return self.bin_op(self.call, (TT_POW, ), self.factor)

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []

            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(' or 'NOT'"
                    ))

                while self.current_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected ',' or ')'"
                    ))

                res.register_advancement()
                self.advance()
            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_MOD))

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, 'not'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected int, float, identifier, '+', '-', '(' or 'not'"
            ))

        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, 'var'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '='"
                ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'VAR', int, float, identifier, '+', '-' or '('"
            ))

        return res.success(node)

    ###################################

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self

    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        # Note: this will allow you to continue and break outside the current function
        return (
            self.error or
            self.func_return_value or
            self.loop_should_continue or
            self.loop_should_break
        )

#######################################
# VALUES
#######################################

class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def powed_by(self, other):
        return None, self.illegal_operation(other)

    def moded_by(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self, other):
        return None, self.illegal_operation(other)

    def execute(self, args):
        return RTResult().failure(self.illegal_operation())

    def copy(self):
        raise Exception('No copy method defined')

    def is_true(self):
        return False

    def illegal_operation(self, other=None):
        if not other: other = self
        return RTError(
            self.pos_start, other.pos_end,
            'Illegal operation',
            self.context
        )

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def moded_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Modulo by zero',
                    self.context
                )

            return Number(self.value % other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)


    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_pi = Number(math.pi)
Number.math_e = Number(math.e)

class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'

class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def added_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def subbed_by(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return None, RTError(
                other.pos_start, other.pos_end,
                'Element at this index could not be removed from list because index is out of bounds',
                self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, List):
            new_list = self.copy()
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except:
                return None, RTError(
                other.pos_start, other.pos_end,
                'Element at this index could not be retrieved from list because index is out of bounds',
                self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'[{", ".join([str(x) for x in self.elements])}]'

class Dictionary(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def copy(self):
        copy = Dictionary(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def dived_by(self, other):
        if isinstance(other, Dictionary):
            return None, RTError(
                other.pos_start, other.pos_end,
                f"You can't use another dictionary as a key",
                self.context
            )

        if isinstance(other, List):
            return None, RTError(
                other.pos_start, other.pos_end,
                f"You can't use a list as a key",
                self.context
            )

        key = other.value

        if key in self.elements:
            return self.elements[key], None
        else:
            return None, RTError(
                other.pos_start, other.pos_end,
                f"Key '{key}' not found in dictionary",
                self.context
            )

    def added_to(self, other):
        if isinstance(other, Dictionary):
            combined_elements = {**self.elements, **other.elements}
            return Dictionary(combined_elements), None
        else:
            return None, Value.illegal_operation(self, other)


    def multed_by(self, other):
        if isinstance(other, Dictionary):
            combined_elements = {**self.elements, **other.elements}
            return Dictionary(combined_elements), None
        else:
            return None, Value.illegal_operation(self, other)

    def subbed_by(self, other):

        if isinstance(other, Dictionary):
            return None, RTError(
                other.pos_start, other.pos_end,
                f"You can't use another dictionary as a key",
                self.context
            )

        if isinstance(other, List):
            return None, RTError(
                other.pos_start, other.pos_end,
                f"You can't use a list as a key",
                self.context
            )

        key = other.value
        if key in self.elements:
            new_elements = self.elements.copy()
            del new_elements[key]
            return Dictionary(new_elements), None
        else:
            return None, RTError(
                other.pos_start, other.pos_end,
                f"Key '{key}' not found in dictionary",
                self.context
            )


    def __str__(self):
        return "{" + ", ".join([f'{key} : {value}' for key, value in self.elements.items()]) + "}"

    def __repr__(self):
        return "{" + ", ".join([f'{repr(key)} : {repr(value)}' for key, value in self.elements.items()]) + "}"



class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_args(self, arg_names, args):
        res = RTResult()

        if len(args) > len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(arg_names)} too many args passed into {self}",
                self.context
            ))

        if len(args) < len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(arg_names) - len(args)} too few args passed into {self}",
                self.context
            ))

        return res.success(None)

    def populate_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)

    def check_and_populate_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.should_return(): return res
        self.populate_args(arg_names, args, exec_ctx)
        return res.success(None)

class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_auto_return = should_auto_return

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_new_context()

        res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
        if res.should_return(): return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return_value == None: return res

        ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res
        return res.success(return_value)

    def no_visit_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    #####################################

    def execute_print(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number.null)
    execute_print.arg_names = ['value']

    def execute_print_ret(self, exec_ctx):
        return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
    execute_print_ret.arg_names = ['value']

    def execute_input(self, exec_ctx):
        text_ = exec_ctx.symbol_table.get('text')
        text = input(text_.value)
        return RTResult().success(String(text))
    execute_input.arg_names = ["text"]

    def execute_input_int(self, exec_ctx):
        text_ = exec_ctx.symbol_table.get('text')
        while True:
            text = input(text_.value)
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RTResult().success(Number(number))
    execute_input_int.arg_names = ["text"]

    def execute_clear(self, exec_ctx):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RTResult().success(Number.null)
    execute_clear.arg_names = []

    def execute_is_number(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
        return RTResult().success(Number.true if is_number else Number.false)
    execute_is_number.arg_names = ["value"]

    def execute_is_string(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
        return RTResult().success(Number.true if is_number else Number.false)
    execute_is_string.arg_names = ["value"]

    def execute_is_list(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
        return RTResult().success(Number.true if is_number else Number.false)
    execute_is_list.arg_names = ["value"]

    def execute_is_function(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
        return RTResult().success(Number.true if is_number else Number.false)
    execute_is_function.arg_names = ["value"]

    def execute_append(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        value = exec_ctx.symbol_table.get("value")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        list_.elements.append(value)
        return RTResult().success(Number.null)
    execute_append.arg_names = ["list", "value"]

    def execute_pop(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        index = exec_ctx.symbol_table.get("index")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        if not isinstance(index, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be number",
                exec_ctx
            ))

        try:
            element = list_.elements.pop(index.value)
        except:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                'Element at this index could not be removed from list because index is out of bounds',
                exec_ctx
            ))
        return RTResult().success(element)
    execute_pop.arg_names = ["list", "index"]

    def execute_extend(self, exec_ctx):
        listA = exec_ctx.symbol_table.get("listA")
        listB = exec_ctx.symbol_table.get("listB")

        if not isinstance(listA, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        if not isinstance(listB, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be list",
                exec_ctx
            ))

        listA.elements.extend(listB.elements)
        return RTResult().success(Number.null)
    execute_extend.arg_names = ["listA", "listB"]

    def execute_check_key(self, exec_ctx):
        dictionary = exec_ctx.symbol_table.get("dictionary")
        key = exec_ctx.symbol_table.get("key")

        if not isinstance(dictionary, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Fist argument must be a dictionary",
                exec_ctx
            ))

        if isinstance(key, (BuiltInFunction, Function, List, Dictionary)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "You can only use numbers or strings as keys",
                exec_ctx
            ))

        if key.value in dictionary.elements:
            return RTResult().success(Number.true)
        else:
            return RTResult().success(Number.false)
    execute_check_key.arg_names = ["dictionary", "key"]

    def execute_merge(self, exec_ctx):
        dictionary1 = exec_ctx.symbol_table.get("dictionary1")
        dictionary2 = exec_ctx.symbol_table.get("dictionary2")

        if not isinstance(dictionary1, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Fist argument must be a dictionary",
                exec_ctx
            ))

        if not isinstance(dictionary2, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be a dictionary",
                exec_ctx
            ))

        combined_elements = {**dictionary1.elements, **dictionary2.elements}
        return RTResult().success(Dictionary(combined_elements))
    execute_merge.arg_names = ["dictionary1", "dictionary2"]

    def execute_remove_key_value_pairs(self, exec_ctx):
        dictionary = exec_ctx.symbol_table.get("dictionary")
        key = exec_ctx.symbol_table.get("key")

        if not isinstance(dictionary, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Fist argument must be a dictionary",
                exec_ctx
            ))

        if isinstance(key, (BuiltInFunction, Function, List, Dictionary)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "You can only use numbers or strings as keys",
                exec_ctx
            ))

        try:
            del dictionary.elements[key.value]
        except KeyError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The key {key.value} is not present in the dictionary",
                exec_ctx
            ))
        return RTResult().success(Dictionary(dictionary.elements))
    execute_remove_key_value_pairs.arg_names = ["dictionary", "key"]

    def execute_get_value_from_key(self, exec_ctx):
        dictionary = exec_ctx.symbol_table.get("dictionary")
        key = exec_ctx.symbol_table.get("key")

        if not isinstance(dictionary, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Fist argument must be a dictionary",
                exec_ctx
            ))

        if isinstance(key, (BuiltInFunction, Function, List, Dictionary)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "You can only use numbers or strings as keys",
                exec_ctx
            ))

        value = dictionary.elements.get(key.value)
        if value is None:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The key {key.value} does not exist in the dictionary",
                exec_ctx
            ))
        return RTResult().success(value)
    execute_get_value_from_key.arg_names = ["dictionary", "key"]

    def execute_sort(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list",
                exec_ctx
            ))

        try:
            list_.elements.sort(key=lambda x: x.value)
        except TypeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Can't sort list with both numbers and strings",
                exec_ctx
            ))
        except AttributeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list '{list_}' can't contain lists or fuctions",
                exec_ctx
            ))

        return RTResult().success(Number.null)
    execute_sort.arg_names = ["list"]

    def execute_length(self, exec_ctx):
        value_ = exec_ctx.symbol_table.get("value")

        if isinstance(value_, String):
            lenght = len(value_.value)
        elif isinstance(value_, List):
            lenght = len(value_.elements)
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list or a string",
                exec_ctx
            ))

        return RTResult().success(Number(lenght))
    execute_length.arg_names = ["value"]

    def execute_smallest_in_list(self, exec_ctx):
        """Get smallest value in a list using python min() function"""
        list_ = exec_ctx.symbol_table.get("list")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list",
                exec_ctx
            ))
        try:
            # get the smallest number from the list
            smallest_number = min(list_.elements, key=lambda x: x.value)
        except ValueError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"List '{list_}' is empty",
                exec_ctx
            ))
        except AttributeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list '{list_}' can't contain lists or fuctions",
                exec_ctx
            ))
        except TypeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list '{list_}' can't contain both strings and numbers",
                exec_ctx
            ))

        return RTResult().success(Number(smallest_number))
    execute_smallest_in_list.arg_names = ["list"]

    def execute_largest_in_list(self, exec_ctx):
        """Get smallest value in a list using python min() function"""
        list_ = exec_ctx.symbol_table.get("list")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list",
                exec_ctx
            ))
        try:
            # get the smallest number from the list
            smallest_number = max(list_.elements, key=lambda x: x.value)
        except ValueError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"List '{list_}' is empty",
                exec_ctx
            ))
        except AttributeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list '{list_}' can't contain lists or fuctions",
                exec_ctx
            ))
        except TypeError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list '{list_}' can't contain both strings and numbers",
                exec_ctx
            ))

        return RTResult().success(Number(smallest_number))
    execute_largest_in_list.arg_names = ["list"]

    def execute_to_string(self, exec_ctx):
        """Turns a number or value into a string"""
        value = exec_ctx.symbol_table.get("value")

        if isinstance(value, Number):
            return RTResult().success(String(str(value.value)))
        elif isinstance(value, List):
            return RTResult().success(String(str(value)))
        elif isinstance(value, String):
            return RTResult().success(value)

        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            f"Can't convert to string '{value}'",
            exec_ctx
        ))
    execute_to_string.arg_names = ["value"]

    def execute_to_number(self, exec_ctx):
        """Turns a value to a number"""
        value = exec_ctx.symbol_table.get("value")
        if isinstance(value, Number):
            return RTResult().success(value)
        elif isinstance(value, String):
            try:
                try:
                    return RTResult().success(Number(int(value.value)))
                except:
                    return RTResult().success(Number(float(value.value)))
            except ValueError:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f"Can't convert to number '{value}'",
                    exec_ctx
                ))
        elif isinstance(value, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Can't convert a list to number",
                exec_ctx
            ))
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Can't convert to number '{value}'",
                exec_ctx
            ))
    execute_to_number.arg_names = ["value"]

    def execute_to_intiger(self, exec_ctx):
        """Turns a value to an intiger"""
        value = exec_ctx.symbol_table.get("value")
        if isinstance(value, Number):
            return RTResult().success(Number(int(value.value)))
        elif isinstance(value, String):
            try:
                try:
                    return RTResult().success(Number(int(value.value)))
                except:
                    float_ = float(value.value)
                    return RTResult().success(Number(int(float_)))
            except ValueError:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f"Can't convert to intiger '{value}'",
                    exec_ctx
                ))
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Can't convert to intiger '{value}'",
                exec_ctx
            ))
    execute_to_intiger.arg_names = ["value"]

    def execute_to_float(self, exec_ctx):
        """Turns a value to an intiger"""
        value = exec_ctx.symbol_table.get("value")
        if isinstance(value, Number):
            return RTResult().success(Number(float(value.value)))
        elif isinstance(value, String):
            try:
                try:
                    return RTResult().success(Number(float(value.value)))
                except:
                    return RTResult().success(Number(float(value.value)))
            except ValueError:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f"Can't convert to float '{value}'",
                    exec_ctx
                ))
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Can't convert to float '{value}'",
                exec_ctx
            ))
    execute_to_float.arg_names = ["value"]

    def execute_absolute_value(self, exec_ctx):
        """Turns a value to an intiger"""
        number_ = exec_ctx.symbol_table.get("number")

        if not isinstance(number_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be number",
                exec_ctx
            ))

        return RTResult().success(Number(abs(number_.value)))
    execute_absolute_value.arg_names = ["number"]

    def execute_range(self, exec_ctx):
        start_ = exec_ctx.symbol_table.get("numberA")
        end_ = exec_ctx.symbol_table.get("numberB")
        step_ = exec_ctx.symbol_table.get("numberC")


        if not isinstance(start_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a number",
                exec_ctx
            ))
        elif not isinstance(end_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be a number",
                exec_ctx
            ))
        elif not isinstance(step_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Third argument must be a number",
                exec_ctx
            ))

        range_ = list(range(start_.value, end_.value, step_.value))

        return RTResult().success(List(range_))
    execute_range.arg_names = ["numberA", "numberB", "numberC"]

    def execute_factorial(self, exec_ctx):
        number_ = exec_ctx.symbol_table.get("number")


        if not isinstance(number_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        if isinstance(number_.value, int):
            factorial_ = math.factorial(number_.value)
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Can only calculate the factorial of an intiger",
                exec_ctx
            ))



        return RTResult().success(Number(factorial_))
    execute_factorial.arg_names = ["number"]

    def execute_raise_error(self, exec_ctx):
        string_ = exec_ctx.symbol_table.get("string")


        if not isinstance(string_, String):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            string_,
            exec_ctx
        ))
    execute_raise_error.arg_names = ["string"]

# TODO fix this function
    def execute_count(self, exec_ctx):
        """Count the elements in a list using the python count() statement"""
        value_ = exec_ctx.symbol_table.get("value")
        element_ = exec_ctx.symbol_table.get("element")

        if not (isinstance(value_, (List, String))):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list or a string",
                exec_ctx
            ))

        if not (isinstance(element_, (Number, String))):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be number or string",
                exec_ctx
            ))

        if isinstance(value_, String):
            try:
                count_ = value_.value.count(element_.value)
            except TypeError:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f'Argument {element_.value} must be a string"',
                    exec_ctx
                ))
        elif isinstance(value_, List):
            count_ = 0

        return RTResult().success(Number(count_))
    execute_count.arg_names = ["value", "element"]

    def execute_random_float(self, exec_ctx):
        start_ = exec_ctx.symbol_table.get("valueA")
        end_ = exec_ctx.symbol_table.get("valueB")

        if not (isinstance(start_, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        if not (isinstance(end_, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        result_ = random.uniform(start_.value, end_.value)

        return RTResult().success(Number(result_))
    execute_random_float.arg_names = ["valueA", "valueB"]

    def execute_random_intiger(self, exec_ctx):
        start_ = exec_ctx.symbol_table.get("valueA")
        end_ = exec_ctx.symbol_table.get("valueB")

        if not (isinstance(start_, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        if not (isinstance(end_, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        try:
            result_ = random.randint(start_.value, end_.value)
        except ValueError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "If first argument is positive second argument can't be negative",
                exec_ctx
            ))

        return RTResult().success(Number(result_))
    execute_random_intiger.arg_names = ["valueA", "valueB"]

    def execute_is_prime(self, exec_ctx):
        number_ = exec_ctx.symbol_table.get("number")

        if not (isinstance(number_, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

        prime_flag = 0

        if (number_.value > 1):
            for i in range(2, int(math.sqrt(number_.value)) + 1):
                if (number_.value % i == 0):
                    prime_flag = 1
                    break
            if (prime_flag == 0):
                return RTResult().success(Number.true)
            else:
                return RTResult().success(Number.false)
        else:
            return RTResult().success(Number.false)
    execute_is_prime.arg_names = ["number"]

    def execute_list_sum(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")

        if not (isinstance(list_, List)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a list",
                exec_ctx
            ))

        try:
            sum_ = sum([x.value for x in list_.elements])
        except TypeError as e:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"The list can only contain numbers",
                exec_ctx
            ))

        return RTResult().success(Number(sum_))
    execute_list_sum.arg_names = ["list"]

    def execute_slice(self, exec_ctx):
        """Slices a list"""
        list_ = exec_ctx.symbol_table.get("list")
        start_ = exec_ctx.symbol_table.get("start")
        end_ = exec_ctx.symbol_table.get("end")

        if not (isinstance(list_, List)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a list",
                exec_ctx
            ))

        if not isinstance(start_.value, int):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be an integer",
                exec_ctx
            ))
        if not isinstance(end_.value, int):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Third argument must be an integer",
                exec_ctx
            ))

        sliced_list = list_.elements[start_.value : end_.value]

        return RTResult().success(List(sliced_list))
    execute_slice.arg_names = ["list", "start", "end"]

    def execute_insert(self, exec_ctx):
        """Adds a new element to a list using the python insert() method"""
        list_ = exec_ctx.symbol_table.get("list")
        index_ = exec_ctx.symbol_table.get("index")
        value_ = exec_ctx.symbol_table.get("value")
        if not (isinstance(list_, List)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a list",
                exec_ctx
            ))

        if not isinstance(index_, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be an integer",
                exec_ctx
            ))
        if not isinstance(value_, Value):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Third argument must be a number or string",
                exec_ctx
            ))

        try:
            list_.elements.insert(index_.value, value_.value)
        except:
            list_.elements.insert(index_.value, value_)

        return RTResult().success(Number.null)
    execute_insert.arg_names = ["list", "index", "value"]

    def execute_run(self, exec_ctx):
        fn = exec_ctx.symbol_table.get("fn")

        if not (isinstance(fn, String)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a string",
                exec_ctx
            ))

        fn = fn.value

        if fn.endswith('.zys'):
            try:
                with open(fn, "r") as f:
                    script = f.read()
            except Exception as e:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f"Failed to load script \"{fn}\"\n" + str(e),
                    exec_ctx
                ))

            _, error = run(fn, script)

            if error:
                return RTResult().failure(RTError(
                    self.pos_start, self.pos_end,
                    f"Failed to finish executing script \"{fn}\"\n" +
                    error.as_string(),
                    exec_ctx
                ))

            return RTResult().success(Number.null)

        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Script file name must ends with \".zys\"\n" +
                f"But got {fn}",
                exec_ctx
            ))

    execute_run.arg_names = ["fn"]

    def execute_quit(self, exec_ctx):
        quit()
    execute_quit.arg_names = []



BuiltInFunction.print                  = BuiltInFunction("print")
BuiltInFunction.print_ret              = BuiltInFunction("print_ret")
BuiltInFunction.input                  = BuiltInFunction("input")
BuiltInFunction.input_int              = BuiltInFunction("input_int")
BuiltInFunction.clear                  = BuiltInFunction("clear")
BuiltInFunction.is_number              = BuiltInFunction("is_number")
BuiltInFunction.is_string              = BuiltInFunction("is_string")
BuiltInFunction.is_list                = BuiltInFunction("is_list")
BuiltInFunction.is_function            = BuiltInFunction("is_function")
BuiltInFunction.is_prime               = BuiltInFunction("is_prime")
BuiltInFunction.merge                  = BuiltInFunction("merge")
BuiltInFunction.remove_key_value_pairs = BuiltInFunction("remove_key_value_pairs")
BuiltInFunction.get_value_from_key     = BuiltInFunction("get_value_from_key")
BuiltInFunction.check_key              = BuiltInFunction("check_key")
BuiltInFunction.slice                  = BuiltInFunction("slice")
BuiltInFunction.insert                 = BuiltInFunction("insert")
BuiltInFunction.append                 = BuiltInFunction("append")
BuiltInFunction.pop                    = BuiltInFunction("pop")
BuiltInFunction.extend                 = BuiltInFunction("extend")
BuiltInFunction.sort                   = BuiltInFunction("sort")
BuiltInFunction.length                 = BuiltInFunction("length")
BuiltInFunction.count                  = BuiltInFunction("count")
BuiltInFunction.to_string              = BuiltInFunction("to_string")
BuiltInFunction.to_number              = BuiltInFunction("to_number")
BuiltInFunction.to_intiger             = BuiltInFunction("to_intiger")
BuiltInFunction.to_float               = BuiltInFunction("to_float")
BuiltInFunction.absolute_value         = BuiltInFunction("absolute_value")
BuiltInFunction.smallest_in_list       = BuiltInFunction("smallest_in_list")
BuiltInFunction.largest_in_list        = BuiltInFunction("largest_in_list")
BuiltInFunction.list_sum               = BuiltInFunction("list_sum")
BuiltInFunction.range                  = BuiltInFunction("range")
BuiltInFunction.factorial              = BuiltInFunction("factorial")
BuiltInFunction.raise_error            = BuiltInFunction("raise_error")
BuiltInFunction.random_float           = BuiltInFunction("random_float")
BuiltInFunction.random_intiger         = BuiltInFunction("random_intiger")
BuiltInFunction.run                    = BuiltInFunction("run")
BuiltInFunction.quit                   = BuiltInFunction("quit")

#######################################
# CONTEXT
#######################################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]

#######################################
# INTERPRETER
#######################################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    ###################################

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )


    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []

        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.should_return(): return res

        return res.success(
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )


    def visit_DictionaryNode(self, node, context):
        res = RTResult()
        elements = {}

        for key_node, value_node in node.key_value_pairs:
            key = res.register(self.visit(key_node, context))
            if res.should_return(): return res

            if isinstance(key, Dictionary):
                return res.failure(RTError(
                    key_node.pos_start, key_node.pos_end,
                    "You can't use another dictionary as a key",
                    context
                ))

            if isinstance(key, List):
                return res.failure(RTError(
                    key_node.pos_start, key_node.pos_end,
                    "You can't use a list as a key",
                    context
                ))

            value = res.register(self.visit(value_node, context))
            if res.should_return(): return res

            elements[key.value] = value

        return res.success(
            Dictionary(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )



    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res

        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TT_MOD:
            result, error = left.moded_by(right)
        elif node.op_tok.type == TT_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TT_KEYWORD, 'and'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, 'or'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return(): return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr, should_return_null in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return(): return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Number.null if should_return_null else expr_value)

        if node.else_case:
            expr, should_return_null = node.else_case
            expr_value = res.register(self.visit(expr, context))
            if res.should_return(): return res
            return res.success(Number.null if should_return_null else expr_value)

        return res.success(Number.null)

    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.should_return(): return res
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res

            if not condition.is_true():
                break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res

        return_value = res.register(value_to_call.execute(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Number.null

        return res.success_return(value)

    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()

    def visit_BreakNode(self, node, context):
        return RTResult().success_break()

    def visit_TryCatchNode(self, node, context):
        res = RTResult()

        try_result = res.register(self.visit(node.try_body, context))
        if res.should_return():
            catch_result = res.register(self.visit(node.catch_body, context))
            if res.should_return(): return res
            return res.success(catch_result)
        return res.success(try_result)

#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)

global_symbol_table.set("math_pi", Number.math_pi)
global_symbol_table.set("math_e", Number.math_e)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("input_int", BuiltInFunction.input_int)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("is_num", BuiltInFunction.is_number)
global_symbol_table.set("is_str", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_function", BuiltInFunction.is_function)
global_symbol_table.set("is_prime", BuiltInFunction.is_prime)
global_symbol_table.set("get_value", BuiltInFunction.get_value_from_key)
global_symbol_table.set("key", BuiltInFunction.check_key)
global_symbol_table.set("merge", BuiltInFunction.merge)
global_symbol_table.set("del", BuiltInFunction.remove_key_value_pairs)
global_symbol_table.set("slice", BuiltInFunction.slice)
global_symbol_table.set("insert", BuiltInFunction.insert)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("sort", BuiltInFunction.sort)
global_symbol_table.set("len", BuiltInFunction.length)
global_symbol_table.set("count", BuiltInFunction.count)
global_symbol_table.set("str", BuiltInFunction.to_string)
global_symbol_table.set("num", BuiltInFunction.to_number)
global_symbol_table.set("int", BuiltInFunction.to_intiger)
global_symbol_table.set("float", BuiltInFunction.to_float)
global_symbol_table.set("abs", BuiltInFunction.absolute_value)
global_symbol_table.set("min", BuiltInFunction.smallest_in_list)
global_symbol_table.set("max", BuiltInFunction.largest_in_list)
global_symbol_table.set("sum", BuiltInFunction.list_sum)
global_symbol_table.set("range", BuiltInFunction.range)
global_symbol_table.set("factorial", BuiltInFunction.factorial)
global_symbol_table.set("raise_error", BuiltInFunction.raise_error)
global_symbol_table.set("randfloat", BuiltInFunction.random_float)
global_symbol_table.set("randint", BuiltInFunction.random_intiger)
global_symbol_table.set("run", BuiltInFunction.run)
global_symbol_table.set("quit", BuiltInFunction.quit)

def run(fn, text):
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error