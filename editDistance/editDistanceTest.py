import editDistance

def test():
    E = editDistance.EditDistance()
    (c, d, s) = E.minEditDistance('execution', 'intention')
    assert c == [1, 3, 1] and d == 8
    (c, d, s) = E.minEditDistance('dog', 'cat')
    assert c == [0, 3, 0] and d == 6
    (c, d, s) = E.minEditDistance('exclusive', 'excusi')
    assert c == [3, 0, 0] and d == 3
    (c, d, s) = E.minEditDistance([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    assert c == [1, 0, 1] and d == 2
    
if __name__ == '__main__':
    test()
