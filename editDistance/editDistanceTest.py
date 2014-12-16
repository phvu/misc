import editDistance

def test():
    E = editDistance.EditDistance()
    (c, d, s) = E.minEditDistanceDebug('execution', 'intention')
    assert c == [1, 3, 1] and d == 8
    assert E.minEditDistance('execution', 'intention') == d
    (c, d, s) = E.minEditDistanceDebug('dog', 'cat')
    assert c == [0, 3, 0] and d == 6
    assert E.minEditDistance('dog', 'cat') == d
    (c, d, s) = E.minEditDistanceDebug('exclusive', 'excusi')
    assert c == [3, 0, 0] and d == 3
    assert E.minEditDistance('exclusive', 'excusi') == d
    (c, d, s) = E.minEditDistanceDebug([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    assert c == [1, 0, 1] and d == 2
    assert E.minEditDistance([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]) == d
    
if __name__ == '__main__':
    test()
