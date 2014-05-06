import ast

class W2vVisitor(ast.NodeVisitor):

    def visit_Module(self, node):
        self.results = None
        self.operands = []
        self.generic_visit(node)
        assert len(self.operands) == 1  # throw exception
        
    def visit_Name(self, node):
        self.operands.append(1)
    
    def visit_Num(self, node):
        self.operands.append(node.n)
    
    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        opType = type(node.op)
        if opType is ast.UAdd:
            pass
        elif opType is ast.USub:
            self.operands.append(-self.operands.pop())
        else:
            raise Exception('Unsupported unary operator: %s' % type(node.op))
            
    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

        opRight = self.operands.pop()
        opLeft = self.operands.pop()
        opType = type(node.op)
        if opType is ast.Add:
            self.operands.append(opLeft + opRight)
        elif opType is ast.Mult:
            self.operands.append(opLeft * opRight)
        elif opType is ast.Sub:
            self.operands.append(opLeft - opRight)
        else:
            raise Exception('Unsupported binary operator: %s' % type(node.op))
        
def test(s):
    v = W2vVisitor()
    v.visit(ast.parse(s))
    print 'Result:', v.operands.pop()

def listItems(s):
    return [x for x in ast.walk(ast.parse(s))]
