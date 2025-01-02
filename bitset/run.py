from bitset import Bitset

if __name__ == '__main__':
    b = Bitset(10)
    b.set(1)
    b.set(5)

    a = Bitset(10)
    a.set(1)

    print(a)
    print(b)
    print(a or b)
    print(a and b)

